//! OpenXR graphics setup for Horizon's Vulkan renderer.
//!
//! OpenXR selects the Vulkan instance and device that its runtime accepts for
//! the headset. This crate creates those handles through `XR_KHR_vulkan_enable2`
//! and wraps them in wgpu-hal so both APIs use the same GPU objects.

use std::error::Error;

use ash::{vk, vk::Handle as _};
use openxr as xr;
use wgpu::hal::{self, api::Vulkan};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

/// An OpenXR-selected Vulkan instance and wgpu adapter.
///
/// The device is opened after Horizon chooses the exact wgpu features and
/// limits it needs. Keep this value alive until the renderer's wgpu device is
/// dropped; OpenXR must outlive the Vulkan objects it selected.
pub struct XrGraphics {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    xr_instance: xr::Instance,
    system: xr::SystemId,
    physical_device: vk::PhysicalDevice,
}

impl XrGraphics {
    /// Opens the active OpenXR runtime and exposes its Vulkan adapter to wgpu.
    pub fn new() -> Result<Self> {
        let entry = unsafe { xr::Entry::load(&())? };
        let available = entry.enumerate_extensions()?;
        if !available.khr_vulkan_enable2 {
            return Err("the active OpenXR runtime does not support XR_KHR_vulkan_enable2".into());
        }

        let mut extensions = xr::ExtensionSet::default();
        extensions.khr_vulkan_enable2 = true;
        let xr_instance = entry.create_instance(
            &xr::ApplicationInfo {
                application_name: "Horizon",
                application_version: 1,
                engine_name: "Horizon",
                engine_version: 1,
                api_version: xr::Version::new(1, 0, 0),
            },
            &extensions,
            &[],
            &(),
        )?;
        let system = xr_instance.system(xr::FormFactor::HEAD_MOUNTED_DISPLAY)?;
        let requirements = xr_instance.graphics_requirements::<xr::Vulkan>(system)?;

        // Vulkan 1.1 is the minimum wgpu-hal needs for the features Horizon
        // uses. The runtime's range is checked before it creates the instance.
        let vk_version = vk::make_api_version(0, 1, 1, 0);
        let xr_version = xr::Version::new(1, 1, 0);
        if xr_version < requirements.min_api_version_supported
            || xr_version > requirements.max_api_version_supported
        {
            return Err(format!(
                "OpenXR requires Vulkan {} through {}, which does not include Vulkan 1.1",
                requirements.min_api_version_supported, requirements.max_api_version_supported
            )
            .into());
        }

        let vk_entry = unsafe { ash::Entry::load()? };
        let flags = wgpu::InstanceFlags::empty();
        let instance_extensions =
            hal::vulkan::Instance::desired_extensions(&vk_entry, vk_version, flags)?;
        let extension_names: Vec<_> = instance_extensions
            .iter()
            .map(|extension| extension.as_ptr())
            .collect();
        let app_info = vk::ApplicationInfo::default()
            .application_version(1)
            .engine_version(1)
            .api_version(vk_version);
        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names);

        let raw_instance = unsafe {
            xr_instance
                .create_vulkan_instance(
                    system,
                    std::mem::transmute(vk_entry.static_fn().get_instance_proc_addr),
                    &create_info as *const _ as *const _,
                )?
                .map_err(vk::Result::from_raw)?
        };
        let ash_instance = unsafe {
            ash::Instance::load(
                vk_entry.static_fn(),
                vk::Instance::from_raw(raw_instance as _),
            )
        };
        let physical_device = unsafe {
            xr_instance.vulkan_graphics_device(system, ash_instance.handle().as_raw() as _)?
        };
        let physical_device = vk::PhysicalDevice::from_raw(physical_device as _);

        // wgpu-hal takes ownership of the Vulkan instance and uses this exact
        // extension list when it queries the runtime-selected physical device.
        let hal_instance = unsafe {
            hal::vulkan::Instance::from_raw(
                vk_entry,
                ash_instance,
                vk_version,
                0,
                None,
                instance_extensions,
                flags,
                wgpu::MemoryBudgetThresholds::default(),
                false,
                None,
            )?
        };
        let exposed = hal_instance
            .expose_adapter(physical_device)
            .ok_or("OpenXR's Vulkan device is not supported by wgpu")?;
        let instance = unsafe { wgpu::Instance::from_hal::<Vulkan>(hal_instance) };
        let adapter = unsafe { instance.create_adapter_from_hal(exposed) };

        Ok(Self {
            instance,
            adapter,
            xr_instance,
            system,
            physical_device,
        })
    }

    pub fn instance(&self) -> &wgpu::Instance {
        &self.instance
    }

    pub fn adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }

    /// Creates a wgpu device through the OpenXR runtime with Horizon's exact
    /// feature and limit request.
    pub fn open_device(
        &self,
        descriptor: &wgpu::DeviceDescriptor<'_>,
    ) -> Result<(wgpu::Device, wgpu::Queue)> {
        let hal_adapter = unsafe { self.adapter.as_hal::<Vulkan>() }
            .ok_or("OpenXR did not provide a Vulkan wgpu adapter")?;
        let extensions = hal_adapter.required_device_extensions(descriptor.required_features);
        let mut physical_features =
            hal_adapter.physical_device_features(&extensions, descriptor.required_features);
        let extension_names: Vec<_> = extensions
            .iter()
            .map(|extension| extension.as_ptr())
            .collect();
        let queue_priorities = [1.0];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(0)
            .queue_priorities(&queue_priorities)];
        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&extension_names);
        let create_info = physical_features.add_to_device_create(create_info);

        let vk_entry = hal_adapter.shared_instance().entry();
        let raw_device = unsafe {
            self.xr_instance
                .create_vulkan_device(
                    self.system,
                    std::mem::transmute(vk_entry.static_fn().get_instance_proc_addr),
                    self.physical_device.as_raw() as _,
                    &create_info as *const _ as *const _,
                )?
                .map_err(vk::Result::from_raw)?
        };
        let ash_device = unsafe {
            ash::Device::load(
                hal_adapter.shared_instance().raw_instance().fp_v1_0(),
                vk::Device::from_raw(raw_device as _),
            )
        };
        let opened = unsafe {
            hal_adapter.device_from_raw(
                ash_device,
                None,
                &extensions,
                descriptor.required_features,
                &descriptor.required_limits,
                &descriptor.memory_hints,
                0,
                0,
            )?
        };
        let (device, queue) = unsafe { self.adapter.create_device_from_hal(opened, descriptor) }?;
        Ok((device, queue))
    }

    /// Returns the OpenXR instance for session creation by the render loop.
    pub fn xr_instance(&self) -> &xr::Instance {
        &self.xr_instance
    }

    pub fn system(&self) -> xr::SystemId {
        self.system
    }

    /// Creates the OpenXR session and its stereo swapchain after wgpu opens
    /// the runtime-selected device.
    pub fn create_session(&self, device: &wgpu::Device) -> Result<XrSession> {
        let hal_device = unsafe { device.as_hal::<Vulkan>() }
            .ok_or("OpenXR did not provide a Vulkan wgpu device")?;
        let (instance, physical_device, raw_device) = {
            let shared = hal_device.shared_instance();
            (
                shared.raw_instance().handle().as_raw() as _,
                self.physical_device.as_raw() as _,
                hal_device.raw_device().handle().as_raw() as _,
            )
        };
        let (session, frame_waiter, frame_stream) = unsafe {
            self.xr_instance.create_session::<xr::Vulkan>(
                self.system,
                &xr::vulkan::SessionCreateInfo {
                    instance,
                    physical_device,
                    device: raw_device,
                    queue_family_index: 0,
                    queue_index: 0,
                },
            )?
        };
        let left_hand = self.xr_instance.string_to_path("/user/hand/left")?;
        let right_hand = self.xr_instance.string_to_path("/user/hand/right")?;
        let action_set = self
            .xr_instance
            .create_action_set("gameplay", "Horizon gameplay", 0)?;
        let left_grip = action_set.create_action::<xr::Posef>(
            "left_grip",
            "Left controller pose",
            &[left_hand],
        )?;
        let right_grip = action_set.create_action::<xr::Posef>(
            "right_grip",
            "Right controller pose",
            &[right_hand],
        )?;
        let left_select = action_set.create_action::<bool>(
            "left_select",
            "Left controller select",
            &[left_hand],
        )?;
        let right_select = action_set.create_action::<bool>(
            "right_select",
            "Right controller select",
            &[right_hand],
        )?;
        let left_stick = action_set.create_action::<xr::Vector2f>(
            "left_stick",
            "Left controller movement",
            &[left_hand],
        )?;
        let right_stick = action_set.create_action::<xr::Vector2f>(
            "right_stick",
            "Right controller turning",
            &[right_hand],
        )?;
        let recenter_action =
            action_set.create_action::<bool>("recenter", "Recenter headset", &[left_hand])?;
        let path = |component: &str| self.xr_instance.string_to_path(component);
        let simple_bindings = [
            xr::Binding::new(&left_grip, path("/user/hand/left/input/grip/pose")?),
            xr::Binding::new(&right_grip, path("/user/hand/right/input/grip/pose")?),
            xr::Binding::new(&left_select, path("/user/hand/left/input/select/click")?),
            xr::Binding::new(&right_select, path("/user/hand/right/input/select/click")?),
            xr::Binding::new(&recenter_action, path("/user/hand/left/input/menu/click")?),
        ];
        let touch_bindings = [
            xr::Binding::new(&left_grip, path("/user/hand/left/input/grip/pose")?),
            xr::Binding::new(&right_grip, path("/user/hand/right/input/grip/pose")?),
            xr::Binding::new(&left_select, path("/user/hand/left/input/trigger/value")?),
            xr::Binding::new(&right_select, path("/user/hand/right/input/trigger/value")?),
            xr::Binding::new(&left_stick, path("/user/hand/left/input/thumbstick")?),
            xr::Binding::new(&right_stick, path("/user/hand/right/input/thumbstick")?),
            xr::Binding::new(
                &recenter_action,
                path("/user/hand/left/input/thumbstick/click")?,
            ),
        ];
        let index_bindings = [
            xr::Binding::new(&left_grip, path("/user/hand/left/input/grip/pose")?),
            xr::Binding::new(&right_grip, path("/user/hand/right/input/grip/pose")?),
            xr::Binding::new(&left_select, path("/user/hand/left/input/trigger/click")?),
            xr::Binding::new(&right_select, path("/user/hand/right/input/trigger/click")?),
            xr::Binding::new(&left_stick, path("/user/hand/left/input/thumbstick")?),
            xr::Binding::new(&right_stick, path("/user/hand/right/input/thumbstick")?),
            xr::Binding::new(
                &recenter_action,
                path("/user/hand/left/input/thumbstick/click")?,
            ),
        ];
        let vive_bindings = [
            xr::Binding::new(&left_grip, path("/user/hand/left/input/grip/pose")?),
            xr::Binding::new(&right_grip, path("/user/hand/right/input/grip/pose")?),
            xr::Binding::new(&left_select, path("/user/hand/left/input/trigger/click")?),
            xr::Binding::new(&right_select, path("/user/hand/right/input/trigger/click")?),
            xr::Binding::new(&left_stick, path("/user/hand/left/input/trackpad")?),
            xr::Binding::new(&right_stick, path("/user/hand/right/input/trackpad")?),
            xr::Binding::new(&recenter_action, path("/user/hand/left/input/menu/click")?),
        ];
        let mixed_reality_bindings = [
            xr::Binding::new(&left_grip, path("/user/hand/left/input/grip/pose")?),
            xr::Binding::new(&right_grip, path("/user/hand/right/input/grip/pose")?),
            xr::Binding::new(&left_select, path("/user/hand/left/input/trigger/value")?),
            xr::Binding::new(&right_select, path("/user/hand/right/input/trigger/value")?),
            xr::Binding::new(&left_stick, path("/user/hand/left/input/thumbstick")?),
            xr::Binding::new(&right_stick, path("/user/hand/right/input/thumbstick")?),
            xr::Binding::new(&recenter_action, path("/user/hand/left/input/menu/click")?),
        ];
        for (profile, bindings) in [
            (
                "/interaction_profiles/khr/simple_controller",
                simple_bindings.as_slice(),
            ),
            (
                "/interaction_profiles/oculus/touch_controller",
                touch_bindings.as_slice(),
            ),
            (
                "/interaction_profiles/valve/index_controller",
                index_bindings.as_slice(),
            ),
            (
                "/interaction_profiles/htc/vive_controller",
                vive_bindings.as_slice(),
            ),
            (
                "/interaction_profiles/microsoft/motion_controller",
                mixed_reality_bindings.as_slice(),
            ),
        ] {
            if let Ok(profile) = path(profile) {
                // Each runtime accepts only profiles and components it knows.
                let _ = self
                    .xr_instance
                    .suggest_interaction_profile_bindings(profile, bindings);
            }
        }
        session.attach_action_sets(&[&action_set])?;
        let left_controller_space =
            left_grip.create_space(&session, left_hand, xr::Posef::IDENTITY)?;
        let right_controller_space =
            right_grip.create_space(&session, right_hand, xr::Posef::IDENTITY)?;
        let reference_space =
            session.create_reference_space(xr::ReferenceSpaceType::LOCAL, xr::Posef::IDENTITY)?;

        let view_sizes = self.xr_instance.enumerate_view_configuration_views(
            self.system,
            xr::ViewConfigurationType::PRIMARY_STEREO,
        )?;
        if view_sizes.len() != 2 {
            return Err(format!(
                "Horizon needs two OpenXR views; this runtime returned {}",
                view_sizes.len()
            )
            .into());
        }
        let size = [
            view_sizes
                .iter()
                .map(|view| view.recommended_image_rect_width)
                .max()
                .unwrap_or(1),
            view_sizes
                .iter()
                .map(|view| view.recommended_image_rect_height)
                .max()
                .unwrap_or(1),
        ];
        let session_formats = session.enumerate_swapchain_formats()?;
        let (format, texture_format) = [
            (vk::Format::B8G8R8A8_UNORM, wgpu::TextureFormat::Bgra8Unorm),
            (vk::Format::R8G8B8A8_UNORM, wgpu::TextureFormat::Rgba8Unorm),
        ]
        .into_iter()
        .find(|(format, _)| session_formats.contains(&(format.as_raw() as _)))
        .ok_or("the OpenXR runtime has no 8-bit color format supported by Horizon")?;
        let swapchain = session.create_swapchain(&xr::SwapchainCreateInfo {
            create_flags: xr::SwapchainCreateFlags::EMPTY,
            usage_flags: xr::SwapchainUsageFlags::COLOR_ATTACHMENT,
            format: format.as_raw() as _,
            sample_count: 1,
            width: size[0],
            height: size[1],
            face_count: 1,
            array_size: 2,
            mip_count: 1,
        })?;
        let blend_modes = self.xr_instance.enumerate_environment_blend_modes(
            self.system,
            xr::ViewConfigurationType::PRIMARY_STEREO,
        )?;
        let blend_mode = blend_modes
            .iter()
            .copied()
            .find(|mode| *mode == xr::EnvironmentBlendMode::OPAQUE)
            .or_else(|| blend_modes.first().copied())
            .ok_or("the OpenXR runtime has no stereo environment blend mode")?;

        // OpenXR events drive session start/stop; waiting for the first frame
        // here would block renderer construction before the event loop runs.
        let event_buffer = xr::EventDataBuffer::new();
        Ok(XrSession {
            swapchain,
            reference_space,
            left_hand,
            right_hand,
            left_controller_space,
            right_controller_space,
            left_grip,
            right_grip,
            left_select,
            right_select,
            left_stick,
            right_stick,
            recenter_action,
            action_set,
            frame_waiter,
            frame_stream,
            session,
            instance: self.xr_instance.clone(),
            event_buffer,
            size,
            format: texture_format,
            blend_mode,
            session_running: false,
            session_focused: false,
            anchor: None,
            recenter_down: false,
        })
    }
}

/// A live OpenXR session with a two-layer stereo swapchain.
pub struct XrSession {
    // Runtime-owned children must be destroyed before their parent session.
    swapchain: xr::Swapchain<xr::Vulkan>,
    reference_space: xr::Space,
    left_hand: xr::Path,
    right_hand: xr::Path,
    left_controller_space: xr::Space,
    right_controller_space: xr::Space,
    left_grip: xr::Action<xr::Posef>,
    right_grip: xr::Action<xr::Posef>,
    left_select: xr::Action<bool>,
    right_select: xr::Action<bool>,
    left_stick: xr::Action<xr::Vector2f>,
    right_stick: xr::Action<xr::Vector2f>,
    recenter_action: xr::Action<bool>,
    action_set: xr::ActionSet,
    frame_waiter: xr::FrameWaiter,
    frame_stream: xr::FrameStream<xr::Vulkan>,
    session: xr::Session<xr::Vulkan>,
    instance: xr::Instance,
    event_buffer: xr::EventDataBuffer,
    size: [u32; 2],
    format: wgpu::TextureFormat,
    blend_mode: xr::EnvironmentBlendMode,
    session_running: bool,
    session_focused: bool,
    anchor: Option<(glam::Vec3, glam::Quat)>,
    recenter_down: bool,
}

/// Tracking and display timing for one OpenXR frame.
pub struct XrFrame {
    pub should_render: bool,
    pub tracking_valid: bool,
    pub views: [xr::View; 2],
    pub image_index: Option<u32>,
    pub input: XrInputState,
    predicted_display_time: xr::Time,
}

/// Standard controller input sampled at the OpenXR predicted display time.
#[derive(Clone, Copy, Debug, Default)]
pub struct XrInputState {
    pub left_controller: Option<XrControllerPose>,
    pub right_controller: Option<XrControllerPose>,
    pub left_select: bool,
    pub right_select: bool,
    pub left_stick: [f32; 2],
    pub right_stick: [f32; 2],
    /// Rising edge recenters the tracked head relative to the game camera.
    pub recenter: bool,
}

/// A tracked controller grip pose in the OpenXR reference space.
#[derive(Clone, Copy, Debug)]
pub struct XrControllerPose {
    pub translation: glam::Vec3,
    pub rotation: glam::Quat,
}

/// One OpenXR eye's view and asymmetric projection matrices.
#[derive(Clone, Copy, Debug)]
pub struct XrEyeMatrices {
    pub view: glam::Mat4,
    pub projection: glam::Mat4,
}

/// Head movement relative to the orientation and position at VR startup.
#[derive(Clone, Copy, Debug)]
pub struct XrHeadPose {
    pub translation: glam::Vec3,
    pub rotation: glam::Quat,
}

impl XrSession {
    pub fn size(&self) -> [u32; 2] {
        self.size
    }

    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// Returns the current head pose relative to the initial center eye.
    pub fn head_pose(&mut self, frame: &XrFrame) -> Option<XrHeadPose> {
        if !frame.tracking_valid {
            return None;
        }
        let left = frame.views[0].pose.position;
        let right = frame.views[1].pose.position;
        let current_position = glam::Vec3::new(
            (left.x + right.x) * 0.5,
            (left.y + right.y) * 0.5,
            (left.z + right.z) * 0.5,
        );
        let orientation = quat(frame.views[0].pose.orientation);
        let (anchor_position, anchor_orientation) =
            *self.anchor.get_or_insert((current_position, orientation));
        Some(XrHeadPose {
            translation: anchor_orientation.inverse() * (current_position - anchor_position),
            rotation: anchor_orientation.inverse() * orientation,
        })
    }

    /// Converts OpenXR's absolute eye poses to camera-relative Horizon view
    /// matrices and asymmetric, reversed-depth projections.
    pub fn eye_matrices(
        &mut self,
        frame: &XrFrame,
        base_view: glam::Mat4,
        far: f32,
    ) -> Option<[XrEyeMatrices; 2]> {
        if !frame.tracking_valid {
            return None;
        }
        let head = self.head_pose(frame)?;
        let left = frame.views[0].pose.position;
        let right = frame.views[1].pose.position;
        let center_position = glam::Vec3::new(
            (left.x + right.x) * 0.5,
            (left.y + right.y) * 0.5,
            (left.z + right.z) * 0.5,
        );
        let anchor_orientation = quat(frame.views[0].pose.orientation) * head.rotation.inverse();
        let anchor_position = center_position - anchor_orientation * head.translation;
        let camera_orientation = base_view.inverse().to_scale_rotation_translation().1;
        // The supplied game view already includes the tracked head rotation
        // for aiming and crosshairs. Remove it here before mapping each runtime
        // eye back into the game's body orientation.
        let body_orientation = camera_orientation * head.rotation.inverse();
        let tracking_to_game = body_orientation * anchor_orientation.inverse();

        Some(std::array::from_fn(|index| {
            let pose = frame.views[index].pose;
            let eye_orientation = tracking_to_game * quat(pose.orientation);
            let eye_position = glam::Vec3::new(pose.position.x, pose.position.y, pose.position.z);
            let eye_offset = tracking_to_game * (eye_position - anchor_position);
            let view = glam::Mat4::from_quat(eye_orientation.inverse())
                * glam::Mat4::from_translation(-eye_offset);
            let projection = xr_projection(frame.views[index].fov, 0.05, far);
            XrEyeMatrices { view, projection }
        }))
    }

    /// Imports the runtime-owned swapchain images as wgpu render targets.
    pub fn import_images(&self, device: &wgpu::Device) -> Result<Vec<wgpu::Texture>> {
        let hal_device = unsafe { device.as_hal::<Vulkan>() }
            .ok_or("OpenXR swapchain images need the Vulkan wgpu device")?;
        let size = wgpu::Extent3d {
            width: self.size[0],
            height: self.size[1],
            depth_or_array_layers: 2,
        };
        let wgpu_descriptor = wgpu::TextureDescriptor {
            label: Some("Horizon OpenXR swapchain"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        };
        let hal_descriptor = hal::TextureDescriptor {
            label: Some("Horizon OpenXR swapchain"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUses::COLOR_TARGET,
            memory_flags: hal::MemoryFlags::empty(),
            view_formats: Vec::new(),
        };
        self.swapchain
            .enumerate_images()?
            .into_iter()
            .map(|image| {
                let image = vk::Image::from_raw(image as _);
                // The runtime owns the image and its memory for the swapchain's
                // lifetime. wgpu tracks only its layout and render usage.
                let hal_texture = unsafe {
                    hal_device.texture_from_raw(
                        image,
                        &hal_descriptor,
                        None,
                        hal::vulkan::TextureMemory::External,
                    )
                };
                Ok(unsafe {
                    device.create_texture_from_hal::<Vulkan>(
                        hal_texture,
                        &wgpu_descriptor,
                        wgpu::TextureUses::UNINITIALIZED,
                    )
                })
            })
            .collect()
    }

    fn sync_input(&mut self, time: xr::Time) -> Result<XrInputState> {
        if !self.session_focused {
            return Ok(XrInputState::default());
        }
        self.session.sync_actions(&[(&self.action_set).into()])?;
        let pose = |space: &xr::Space| -> Result<Option<XrControllerPose>> {
            let location = space.locate(&self.reference_space, time)?;
            let valid =
                xr::SpaceLocationFlags::POSITION_VALID | xr::SpaceLocationFlags::ORIENTATION_VALID;
            if !location.location_flags.contains(valid) {
                return Ok(None);
            }
            let p = location.pose.position;
            Ok(Some(XrControllerPose {
                translation: glam::Vec3::new(p.x, p.y, p.z),
                rotation: quat(location.pose.orientation),
            }))
        };
        let left_select = self.left_select.state(&self.session, self.left_hand)?;
        let right_select = self.right_select.state(&self.session, self.right_hand)?;
        let left_stick = self.left_stick.state(&self.session, self.left_hand)?;
        let right_stick = self.right_stick.state(&self.session, self.right_hand)?;
        let recenter = self.recenter_action.state(&self.session, self.left_hand)?;
        let recenter_down = recenter.is_active && recenter.current_state;
        let recenter_pressed = recenter_down && !self.recenter_down;
        if recenter_pressed {
            self.anchor = None;
        }
        self.recenter_down = recenter_down;
        Ok(XrInputState {
            left_controller: if self.left_grip.is_active(&self.session, self.left_hand)? {
                pose(&self.left_controller_space)?
            } else {
                None
            },
            right_controller: if self.right_grip.is_active(&self.session, self.right_hand)? {
                pose(&self.right_controller_space)?
            } else {
                None
            },
            left_select: left_select.is_active && left_select.current_state,
            right_select: right_select.is_active && right_select.current_state,
            left_stick: if left_stick.is_active {
                [left_stick.current_state.x, left_stick.current_state.y]
            } else {
                [0.0; 2]
            },
            right_stick: if right_stick.is_active {
                [right_stick.current_state.x, right_stick.current_state.y]
            } else {
                [0.0; 2]
            },
            recenter: recenter_pressed,
        })
    }

    /// Begins a frame when the runtime has put the session in the READY state.
    pub fn begin_frame(&mut self) -> Result<Option<XrFrame>> {
        while let Some(event) = self.instance.poll_event(&mut self.event_buffer)? {
            match event {
                xr::Event::SessionStateChanged(event) => {
                    self.session_focused = event.state() == xr::SessionState::FOCUSED;
                    match event.state() {
                        xr::SessionState::READY if !self.session_running => {
                            self.session
                                .begin(xr::ViewConfigurationType::PRIMARY_STEREO)?;
                            self.session_running = true;
                        }
                        xr::SessionState::STOPPING if self.session_running => {
                            self.session.end()?;
                            self.session_running = false;
                        }
                        xr::SessionState::EXITING | xr::SessionState::LOSS_PENDING => {
                            self.session_running = false;
                            return Ok(None);
                        }
                        _ => {}
                    }
                }
                xr::Event::InstanceLossPending(_) => {
                    self.session_running = false;
                    return Ok(None);
                }
                _ => {}
            }
        }
        if !self.session_running {
            return Ok(None);
        }

        let state = self.frame_waiter.wait()?;
        self.frame_stream.begin()?;
        let input = self.sync_input(state.predicted_display_time)?;
        if !state.should_render {
            return Ok(Some(XrFrame {
                should_render: false,
                tracking_valid: false,
                views: [xr::View::default(); 2],
                image_index: None,
                input,
                predicted_display_time: state.predicted_display_time,
            }));
        }
        let (view_flags, views) = self.session.locate_views(
            xr::ViewConfigurationType::PRIMARY_STEREO,
            state.predicted_display_time,
            &self.reference_space,
        )?;
        if views.len() != 2 {
            return Err("the OpenXR runtime stopped returning two stereo views".into());
        }
        let image_index = self.swapchain.acquire_image()?;
        self.swapchain.wait_image(xr::Duration::INFINITE)?;
        Ok(Some(XrFrame {
            should_render: true,
            tracking_valid: view_flags.contains(xr::ViewStateFlags::ORIENTATION_VALID)
                && view_flags.contains(xr::ViewStateFlags::POSITION_VALID),
            views: [views[0], views[1]],
            image_index: Some(image_index),
            input,
            predicted_display_time: state.predicted_display_time,
        }))
    }

    /// Submits both stereo views after the caller has finished GPU work.
    pub fn end_frame(&mut self, frame: XrFrame, rendered: bool) -> Result<()> {
        if frame.image_index.is_some() {
            self.swapchain.release_image()?;
        }
        if rendered && frame.should_render {
            let extent = || xr::Extent2Di {
                width: self.size[0] as i32,
                height: self.size[1] as i32,
            };
            let projection_views = [0, 1].map(|index| {
                xr::CompositionLayerProjectionView::new()
                    .pose(frame.views[index].pose)
                    .fov(frame.views[index].fov)
                    .sub_image(
                        xr::SwapchainSubImage::new()
                            .swapchain(&self.swapchain)
                            .image_array_index(index as u32)
                            .image_rect(xr::Rect2Di {
                                offset: xr::Offset2Di { x: 0, y: 0 },
                                extent: extent(),
                            }),
                    )
            });
            let projection = xr::CompositionLayerProjection::new()
                .space(&self.reference_space)
                .views(&projection_views);
            self.frame_stream.end(
                frame.predicted_display_time,
                self.blend_mode,
                &[&projection],
            )?;
        } else {
            self.frame_stream
                .end(frame.predicted_display_time, self.blend_mode, &[])?;
        }
        Ok(())
    }
}

fn quat(orientation: xr::Quaternionf) -> glam::Quat {
    glam::Quat::from_xyzw(orientation.x, orientation.y, orientation.z, orientation.w).normalize()
}

fn xr_projection(fov: xr::Fovf, near: f32, far: f32) -> glam::Mat4 {
    let left = fov.angle_left.tan();
    let right = fov.angle_right.tan();
    let down = fov.angle_down.tan();
    let up = fov.angle_up.tan();
    let x_scale = 2.0 / (right - left);
    let y_scale = 2.0 / (up - down);
    let x_offset = (right + left) / (right - left);
    let y_offset = (up + down) / (up - down);
    let depth = near / (far - near);
    glam::Mat4::from_cols(
        glam::Vec4::new(x_scale, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, y_scale, 0.0, 0.0),
        glam::Vec4::new(x_offset, y_offset, depth, -1.0),
        glam::Vec4::new(0.0, 0.0, near * far / (far - near), 0.0),
    )
}
