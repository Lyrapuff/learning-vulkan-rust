use ash::vk;
use ash::vk::PFN_vkDebugUtilsMessengerCallbackEXT;

pub struct EngineDebug {
    pub loader: ash::extensions::ext::DebugUtils,
    pub messenger: vk::DebugUtilsMessengerEXT,
}

impl EngineDebug {
    pub fn init(
        entry: &ash::Entry,
        instance: &ash::Instance,
        callback: PFN_vkDebugUtilsMessengerCallbackEXT
    ) -> Result<EngineDebug, vk::Result> {
        let debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
//                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            )
            .pfn_user_callback(callback);

        let loader = ash::extensions::ext::DebugUtils::new(entry, instance);

        let messenger = unsafe {
            loader.create_debug_utils_messenger(&debug_create_info, None)?
        };

        Ok(EngineDebug {
            loader,
            messenger
        })
    }
}

impl Drop for EngineDebug {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_debug_utils_messenger(self.messenger, None)
        }
    }
}