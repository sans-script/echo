use crate::tools::registry::{ToolDefinition, ToolRegistry};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

const IGNORE: &[&str] = &[
    "__pycache__",
    "node_modules",
    ".git",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
];
const SYSTEM: &[&str] = &[
    "etc", "root", "sys", "proc", "dev", "var", "bin", "usr", "boot", "home",
];

#[derive(Debug)]
pub struct FilesystemSandbox {
    root: PathBuf,
}
impl FilesystemSandbox {
    pub fn new(root: impl AsRef<Path>) -> Result<Self, String> {
        fs::create_dir_all(root.as_ref()).map_err(|e| e.to_string())?;
        Ok(Self {
            root: root.as_ref().canonicalize().map_err(|e| e.to_string())?,
        })
    }
    fn within(&self, p: &Path) -> bool {
        p.starts_with(&self.root)
    }
    fn rel(&self, p: &Path) -> String {
        let r = p.strip_prefix(&self.root).unwrap_or(Path::new("."));
        if r.as_os_str().is_empty() {
            "./".into()
        } else {
            r.to_string_lossy().replace('\\', "/")
        }
    }
    pub fn resolve_path(&self, input: &str) -> Result<(PathBuf, String), String> {
        if input.trim().is_empty() {
            return Err("Path cannot be empty.".into());
        }
        let c = input.trim();
        let p = Path::new(c);
        if !p.is_absolute()
            && p.components()
                .any(|component| component == std::path::Component::ParentDir)
        {
            return Err(format!(
                "Access denied: Path '{c}' traverses outside workspace sandbox."
            ));
        }
        if p.is_absolute() {
            let q = p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
            if self.within(&q) {
                return Ok((q.clone(), self.rel(&q)));
            }
            let first = p
                .components()
                .nth(1)
                .and_then(|x| match x {
                    std::path::Component::Normal(v) => v.to_str(),
                    _ => None,
                })
                .unwrap_or("");
            if SYSTEM.contains(&first) {
                return Err(format!(
                    "Access denied: Path '{c}' attempts to access system directory outside workspace."
                ));
            }
            let q = self.root.join(c.trim_start_matches(['/', '\\']));
            let q = q.canonicalize().unwrap_or(q);
            if !self.within(&q) {
                return Err(format!(
                    "Access denied: Path '{c}' resolves outside workspace sandbox."
                ));
            }
            return Ok((q.clone(), self.rel(&q)));
        }
        let q = self.root.join(c);
        let q = q.canonicalize().unwrap_or(q);
        if !self.within(&q) {
            return Err(format!(
                "Access denied: Path '{c}' traverses outside workspace sandbox."
            ));
        }
        Ok((q.clone(), self.rel(&q)))
    }
    fn hint(&self, p: &Path) -> String {
        let n = p
            .file_name()
            .and_then(|x| x.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let mut h = Vec::new();
        if let Some(parent) = p.parent() {
            if let Ok(es) = fs::read_dir(parent) {
                for e in es.flatten().take(100) {
                    let x = e.file_name().to_string_lossy().to_string();
                    if x.to_ascii_lowercase() == n {
                        h.push(self.rel(&e.path()))
                    }
                }
            }
        }
        if h.is_empty() {
            " Use list_directory or tree to see which files exist.".into()
        } else {
            format!(" Did you mean: {}?", h.join(", "))
        }
    }
    pub fn write_file(&self, path: &str, content: &str) -> Result<String, String> {
        let (p, r) = self.resolve_path(path)?;
        if let Some(x) = p.parent() {
            fs::create_dir_all(x).map_err(|e| e.to_string())?
        }
        fs::write(&p, content).map_err(|e| format!("Could not write '{r}': {e}"))?;
        Ok(format!(
            "Successfully wrote {} bytes to '{r}'. Location: {}",
            content.len(),
            p.display()
        ))
    }
    pub fn read_file(&self, path: &str, max: usize) -> Result<String, String> {
        let (p, r) = self.resolve_path(path)?;
        if !p.exists() {
            return Err(format!("File '{r}' does not exist.{}", self.hint(&p)));
        }
        if p.is_dir() {
            return Err(format!(
                "'{r}' is a directory, not a file. Use list_directory instead."
            ));
        }
        let b = fs::read(&p).map_err(|e| e.to_string())?;
        let size = b.len();
        let mut s = String::from_utf8_lossy(&b[..size.min(max)]).into_owned();
        if size > max {
            s += &format!("\n... [File truncated: showing first {max} of {size} bytes]")
        }
        Ok(s)
    }
    pub fn list_directory(&self, path: &str) -> Result<String, String> {
        let (p, r) = self.resolve_path(path)?;
        if !p.exists() {
            return Err(format!("Directory '{r}' does not exist.{}", self.hint(&p)));
        }
        if !p.is_dir() {
            return Err(format!(
                "'{r}' is a file, not a directory. Use read_file instead."
            ));
        }
        let mut es = fs::read_dir(&p)
            .map_err(|e| e.to_string())?
            .flatten()
            .collect::<Vec<_>>();
        es.sort_by_key(|e| {
            (
                e.path().is_file(),
                e.file_name().to_string_lossy().to_ascii_lowercase(),
            )
        });
        if es.is_empty() {
            return Ok(format!("(Directory '{r}' is empty)"));
        }
        let mut o = vec![format!("Contents of '{r}':")];
        for e in es {
            let q = e.path();
            let n = e.file_name().to_string_lossy().to_string();
            if q.is_dir() {
                o.push(format!("  [DIR]  {n}/"))
            } else {
                o.push(format!(
                    "  [FILE] {n} ({} bytes)",
                    fs::metadata(q).map(|m| m.len()).unwrap_or(0)
                ))
            }
        }
        Ok(o.join("\n"))
    }
    pub fn edit_file(&self, path: &str, old: &str, new: &str) -> Result<String, String> {
        let (p, r) = self.resolve_path(path)?;
        if !p.exists() {
            return Err(format!("File '{r}' does not exist.{}", self.hint(&p)));
        }
        if p.is_dir() {
            return Err(format!("'{r}' is a directory, not a file."));
        }
        if old.is_empty() {
            return Err(
                "old_text cannot be empty. Use write_file to create or overwrite a file.".into(),
            );
        }
        let text = String::from_utf8(fs::read(&p).map_err(|e| e.to_string())?)
            .map_err(|_| format!("'{r}' is not a UTF-8 text file."))?;
        let n = text.matches(old).count();
        if n == 0 {
            return Err(format!(
                "old_text was not found in '{r}'. Use read_file and copy the exact text, including spaces and indentation."
            ));
        }
        if n > 1 {
            return Err(format!(
                "old_text appears {n} times in '{r}'. Include more surrounding lines so it matches exactly once."
            ));
        }
        fs::write(&p, text.replacen(old, new, 1)).map_err(|e| e.to_string())?;
        Ok(format!(
            "Edited '{r}': replaced 1 occurrence ({} -> {} chars). Location: {}",
            old.len(),
            new.len(),
            p.display()
        ))
    }
    pub fn tree(
        &self,
        path: &str,
        depth: i64,
        hidden: bool,
        sizes: bool,
    ) -> Result<String, String> {
        let (p, r) = self.resolve_path(path)?;
        if !p.exists() {
            return Err(format!("Directory '{r}' does not exist.{}", self.hint(&p)));
        }
        if !p.is_dir() {
            return Err(format!(
                "'{r}' is a file, not a directory. Use read_file instead."
            ));
        }
        let depth = depth.clamp(1, 6) as usize;
        let mut o: Vec<String> = vec![if r == "./" { ".".to_string() } else { r }];
        let (mut d, mut f) = (0, 0);
        let mut tr = false;
        self.walk(
            &p, "", 1, depth, hidden, sizes, &mut o, &mut d, &mut f, &mut tr,
        );
        o.push(format!(
            "\n{d} directories, {f} files ({}{})",
            if tr {
                "truncated".to_string()
            } else {
                "depth ".to_string()
            },
            if tr { String::new() } else { depth.to_string() }
        ));
        Ok(o.join("\n"))
    }
    fn walk(
        &self,
        p: &Path,
        pre: &str,
        level: usize,
        max: usize,
        hidden: bool,
        sizes: bool,
        o: &mut Vec<String>,
        d: &mut usize,
        f: &mut usize,
        tr: &mut bool,
    ) {
        let Ok(rd) = fs::read_dir(p) else {
            o.push(format!("{pre}└── [access denied]"));
            return;
        };
        let mut es = rd.flatten().collect::<Vec<_>>();
        es.sort_by_key(|e| {
            (
                e.path().is_file(),
                e.file_name().to_string_lossy().to_ascii_lowercase(),
            )
        });
        es.retain(|e| {
            let n = e.file_name().to_string_lossy().to_string();
            !IGNORE.contains(&n.as_str()) && (hidden || !n.starts_with('.'))
        });
        for (i, e) in es.iter().enumerate() {
            if *d + *f >= 200 {
                *tr = true;
                return;
            }
            let last = i + 1 == es.len();
            let b = if last { "└── " } else { "├── " };
            let n = e.file_name().to_string_lossy().to_string();
            let q = e.path();
            if q.is_dir() {
                *d += 1;
                o.push(format!("{pre}{b}{n}/"));
                if level < max && !q.is_symlink() {
                    self.walk(
                        &q,
                        &format!("{pre}{}", if last { "    " } else { "│   " }),
                        level + 1,
                        max,
                        hidden,
                        sizes,
                        o,
                        d,
                        f,
                        tr,
                    )
                }
            } else {
                *f += 1;
                let z = if sizes {
                    format!(
                        " ({} bytes)",
                        fs::metadata(&q).map(|m| m.len()).unwrap_or(0)
                    )
                } else {
                    String::new()
                };
                o.push(format!("{pre}{b}{n}{z}"))
            }
        }
    }
    pub fn search_files(
        &self,
        q: &str,
        path: &str,
        glob: &str,
        max: i64,
        ignore: bool,
    ) -> Result<String, String> {
        if q.is_empty() {
            return Err("query cannot be empty.".into());
        }
        let (p, r) = self.resolve_path(path)?;
        if !p.exists() {
            return Err(format!("Path '{r}' does not exist.{}", self.hint(&p)));
        }
        let max = max.clamp(1, 100) as usize;
        let needle = if ignore {
            q.to_ascii_lowercase()
        } else {
            q.into()
        };
        let mut hits = Vec::new();
        let mut scanned = 0;
        for file in collect(&p, glob) {
            if scanned >= 2000 {
                break;
            }
            let Ok(b) = fs::read(&file) else { continue };
            if b.len() > 1_000_000 || b.iter().take(1024).any(|x| *x == 0) {
                continue;
            }
            scanned += 1;
            for (i, line) in String::from_utf8_lossy(&b).lines().enumerate() {
                let h = if ignore {
                    line.to_ascii_lowercase()
                } else {
                    line.into()
                };
                if h.contains(&needle) {
                    hits.push(format!("{}:{}: {}", self.rel(&file), i + 1, line.trim()));
                    if hits.len() >= max {
                        break;
                    }
                }
            }
            if hits.len() >= max {
                break;
            }
        }
        if hits.is_empty() {
            Ok(format!(
                "No matches for '{q}' in {scanned} file(s) under '{r}'."
            ))
        } else {
            Ok(format!(
                "{} match(es){} under '{r}':\n{}",
                hits.len(),
                if hits.len() >= max {
                    " (limit reached)"
                } else {
                    ""
                },
                hits.join("\n")
            ))
        }
    }
    pub fn find_files(&self, pat: &str, path: &str, max: i64) -> Result<String, String> {
        if pat.trim().is_empty() {
            return Err("pattern cannot be empty.".into());
        }
        let (p, r) = self.resolve_path(path)?;
        if !p.exists() {
            return Err(format!("Directory '{r}' does not exist.{}", self.hint(&p)));
        }
        if !p.is_dir() {
            return Err(format!("'{r}' is a file, not a directory."));
        }
        let max = max.clamp(1, 200) as usize;
        let pat = if pat.chars().any(|c| "*?[".contains(c)) {
            pat.to_ascii_lowercase()
        } else {
            format!("*{}*", pat.to_ascii_lowercase())
        };
        let mut stack = vec![p];
        let mut hits = Vec::new();
        while let Some(d) = stack.pop() {
            let Ok(es) = fs::read_dir(d) else { continue };
            for e in es.flatten() {
                let q = e.path();
                let n = e.file_name().to_string_lossy().to_string();
                if IGNORE.contains(&n.as_str()) || n.starts_with('.') {
                    continue;
                }
                if wild(&pat, &n.to_ascii_lowercase()) {
                    hits.push(format!(
                        "{}{}",
                        self.rel(&q),
                        if q.is_dir() { "/" } else { "" }
                    ));
                    if hits.len() >= max {
                        break;
                    }
                }
                if q.is_dir() && !q.is_symlink() {
                    stack.push(q)
                }
            }
            if hits.len() >= max {
                break;
            }
        }
        if hits.is_empty() {
            Ok(format!("No files or folders matching '{pat}' under '{r}'."))
        } else {
            Ok(format!(
                "{} result(s){} for '{pat}' under '{r}':\n{}",
                hits.len(),
                if hits.len() >= max {
                    " (limit reached)"
                } else {
                    ""
                },
                hits.join("\n")
            ))
        }
    }
}
fn collect(root: &Path, glob: &str) -> Vec<PathBuf> {
    let mut o = Vec::new();
    let mut s = vec![root.to_path_buf()];
    while let Some(d) = s.pop() {
        if d.is_file() {
            o.push(d);
            continue;
        }
        let Ok(es) = fs::read_dir(d) else { continue };
        for e in es.flatten() {
            let q = e.path();
            let n = e.file_name().to_string_lossy().to_string();
            if IGNORE.contains(&n.as_str()) || n.starts_with('.') {
                continue;
            }
            if q.is_dir() {
                s.push(q)
            } else if glob == "*" || wild(&glob.to_ascii_lowercase(), &n.to_ascii_lowercase()) {
                o.push(q)
            }
        }
    }
    o.sort();
    o
}
fn wild(p: &str, t: &str) -> bool {
    let (p, t) = (p.as_bytes(), t.as_bytes());
    let mut d = vec![vec![false; t.len() + 1]; p.len() + 1];
    d[0][0] = true;
    for i in 1..=p.len() {
        if p[i - 1] == b'*' {
            d[i][0] = d[i - 1][0]
        }
        for j in 1..=t.len() {
            d[i][j] = match p[i - 1] {
                b'*' => d[i - 1][j] || d[i][j - 1],
                b'?' => d[i - 1][j - 1],
                c => d[i - 1][j - 1] && c == t[j - 1],
            }
        }
    }
    d[p.len()][t.len()]
}
pub fn register_filesystem_tools(
    reg: &mut ToolRegistry,
    root: impl AsRef<Path>,
) -> Result<Arc<FilesystemSandbox>, String> {
    let fs = Arc::new(FilesystemSandbox::new(root)?);
    let x = Arc::clone(&fs);
    reg.register(ToolDefinition::new("read_file","Read a text file within the workspace.",json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}),false,Arc::new(move|a: &serde_json::Map<String, Value>|x.read_file(a["path"].as_str().unwrap_or(""),65536))))?;
    let x = Arc::clone(&fs);
    reg.register(ToolDefinition::new("write_file","Write text to a file within the workspace.",json!({"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"],"additionalProperties":false}),true,Arc::new(move|a: &serde_json::Map<String, Value>|x.write_file(a["path"].as_str().unwrap_or(""),a["content"].as_str().unwrap_or("")))))?;
    let x = Arc::clone(&fs);
    reg.register(ToolDefinition::new("edit_file","Replace exactly one occurrence in a UTF-8 text file.",json!({"type":"object","properties":{"path":{"type":"string"},"old_text":{"type":"string"},"new_text":{"type":"string"}},"required":["path","old_text","new_text"],"additionalProperties":false}),true,Arc::new(move|a: &serde_json::Map<String, Value>|x.edit_file(a["path"].as_str().unwrap_or(""),a["old_text"].as_str().unwrap_or(""),a["new_text"].as_str().unwrap_or("")))))?;
    let x = Arc::clone(&fs);
    reg.register(ToolDefinition::new("list_directory","List files and directories.",json!({"type":"object","properties":{"path":{"type":"string"}},"additionalProperties":false}),false,Arc::new(move|a: &serde_json::Map<String, Value>|x.list_directory(a.get("path").and_then(Value::as_str).unwrap_or(".")))))?;
    let x = Arc::clone(&fs);
    reg.register(ToolDefinition::new("tree","Show a directory tree.",json!({"type":"object","properties":{"path":{"type":"string"},"max_depth":{"type":"integer"},"show_hidden":{"type":"boolean"},"show_sizes":{"type":"boolean"}},"additionalProperties":false}),false,Arc::new(move|a: &serde_json::Map<String, Value>|x.tree(a.get("path").and_then(Value::as_str).unwrap_or("."),a.get("max_depth").and_then(Value::as_i64).unwrap_or(2),a.get("show_hidden").and_then(Value::as_bool).unwrap_or(false),a.get("show_sizes").and_then(Value::as_bool).unwrap_or(false)))))?;
    let x = Arc::clone(&fs);
    reg.register(ToolDefinition::new("search_files","Search text inside workspace files.",json!({"type":"object","properties":{"query":{"type":"string"},"path":{"type":"string"},"glob":{"type":"string"},"max_results":{"type":"integer"},"ignore_case":{"type":"boolean"}},"required":["query"],"additionalProperties":false}),false,Arc::new(move|a: &serde_json::Map<String, Value>|x.search_files(a["query"].as_str().unwrap_or(""),a.get("path").and_then(Value::as_str).unwrap_or("."),a.get("glob").and_then(Value::as_str).unwrap_or("*"),a.get("max_results").and_then(Value::as_i64).unwrap_or(30),a.get("ignore_case").and_then(Value::as_bool).unwrap_or(true)))))?;
    let x = Arc::clone(&fs);
    reg.register(ToolDefinition::new("find_files","Find files and folders by name.",json!({"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"},"max_results":{"type":"integer"}},"required":["pattern"],"additionalProperties":false}),false,Arc::new(move|a: &serde_json::Map<String, Value>|x.find_files(a["pattern"].as_str().unwrap_or(""),a.get("path").and_then(Value::as_str).unwrap_or("."),a.get("max_results").and_then(Value::as_i64).unwrap_or(50)))))?;
    Ok(fs)
}
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    #[test]
    fn sandbox_rejects_traversal() {
        let d = tempdir().unwrap();
        let f = FilesystemSandbox::new(d.path()).unwrap();
        assert!(f.resolve_path("../outside").is_err())
    }
    #[test]
    fn roundtrip() {
        let d = tempdir().unwrap();
        let f = FilesystemSandbox::new(d.path()).unwrap();
        f.write_file("a/b.txt", "hello world").unwrap();
        assert_eq!(f.read_file("a/b.txt", 100).unwrap(), "hello world");
        f.edit_file("a/b.txt", "world", "rust").unwrap();
        assert_eq!(f.read_file("a/b.txt", 100).unwrap(), "hello rust")
    }
    #[test]
    fn tree_depth() {
        let d = tempdir().unwrap();
        let f = FilesystemSandbox::new(d.path()).unwrap();
        f.write_file("a/b/c.txt", "x").unwrap();
        let s = f.tree(".", 2, false, false).unwrap();
        assert!(s.contains("a/"));
        assert!(s.contains("b/"));
        assert!(!s.contains("c.txt"))
    }
    #[test]
    fn search_find() {
        let d = tempdir().unwrap();
        let f = FilesystemSandbox::new(d.path()).unwrap();
        f.write_file("src/main.rs", "fn main() {}").unwrap();
        assert!(
            f.search_files("main", ".", "*.rs", 30, true)
                .unwrap()
                .contains("src/main.rs:1")
        );
        assert!(
            f.find_files("main.rs", ".", 50)
                .unwrap()
                .contains("src/main.rs")
        )
    }
}
