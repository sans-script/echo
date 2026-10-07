pub mod filesystem;
pub mod registry;
pub mod shell;

use std::path::Path;

/// Registers every tool Echo offers the model, bound to `workspace`.
pub fn register_all(registry: &mut registry::ToolRegistry, workspace: &Path) -> Result<(), String> {
    filesystem::register_filesystem_tools(registry, workspace)?;
    shell::register_shell_tools(registry, workspace)
}
