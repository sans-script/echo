use std::path::{Path, PathBuf};

/// Drops the `\\?\` prefix Windows adds to canonical paths, so they display
/// as typed (`C:\...`). UNC paths are left untouched.
pub fn display_path(path: PathBuf) -> PathBuf {
    let text = path.to_string_lossy();
    match text.strip_prefix(r"\\?\") {
        Some(rest) if !rest.starts_with("UNC\\") => PathBuf::from(rest),
        _ => path,
    }
}

pub fn tree_workspace(workspace: &Path, max_depth: usize) -> String {
    fn walk(path: &Path, prefix: &str, depth: usize, max_depth: usize, output: &mut String) {
        let mut entries = match std::fs::read_dir(path) {
            Ok(entries) => entries.filter_map(Result::ok).collect::<Vec<_>>(),
            Err(_) => return,
        };

        entries.sort_by_key(|entry| entry.file_name().to_string_lossy().to_lowercase());

        for (index, entry) in entries.iter().enumerate() {
            let last = index + 1 == entries.len();
            let branch = if last { "└── " } else { "├── " };
            let child_prefix = if last { "    " } else { "│   " };
            let name = entry.file_name().to_string_lossy().into_owned();

            output.push_str(prefix);
            output.push_str(branch);
            output.push_str(&name);
            output.push('\n');

            if depth < max_depth {
                if let Ok(file_type) = entry.file_type() {
                    if file_type.is_dir() {
                        walk(
                            &entry.path(),
                            &format!("{prefix}{child_prefix}"),
                            depth + 1,
                            max_depth,
                            output,
                        );
                    }
                }
            }
        }
    }

    if !workspace.exists() {
        return format!("Workspace does not exist: {}", workspace.display());
    }

    let mut output = String::from(".\n");
    walk(workspace, "", 0, max_depth, &mut output);
    output.trim_end().into()
}
