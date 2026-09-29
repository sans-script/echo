use std::path::Path;

pub fn list_workspace(workspace: &Path) -> String {
    let mut entries = match std::fs::read_dir(workspace) {
        Ok(entries) => entries.filter_map(Result::ok).collect::<Vec<_>>(),
        Err(err) => return format!("Error listing '{}': {}", workspace.display(), err),
    };

    entries.sort_by_key(|entry| entry.file_name().to_string_lossy().to_lowercase());

    if entries.is_empty() {
        return "(empty)".into();
    }

    let mut output = String::new();

    for entry in entries {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();

        match entry.file_type() {
            Ok(file_type) if file_type.is_dir() => {
                output.push_str(&format!("[DIR]  {name}\n"));
            }
            Ok(_) => {
                let size = std::fs::metadata(&path)
                    .map(|metadata| metadata.len())
                    .unwrap_or(0);
                output.push_str(&format!("[FILE] {name} ({size} bytes)\n"));
            }
            Err(_) => output.push_str(&format!("[?]    {name}\n")),
        }
    }

    output.trim_end().into()
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
