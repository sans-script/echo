//! Lightweight syntax highlighting for fenced code blocks in the model's
//! replies. It is a small per-language tokenizer (keywords, strings,
//! comments, numbers, types, calls, annotations), not a full parser, which
//! keeps Echo free of heavy highlighting dependencies.

use ratatui::{
    style::{Color, Modifier, Style},
    text::Span,
};

const PLAIN: Color = Color::Rgb(171, 178, 191);
const KEYWORD: Color = Color::Rgb(198, 120, 221);
const TYPE: Color = Color::Rgb(229, 192, 123);
const STRING: Color = Color::Rgb(152, 195, 121);
const NUMBER: Color = Color::Rgb(209, 154, 102);
const COMMENT: Color = Color::Rgb(106, 115, 130);
const FUNCTION: Color = Color::Rgb(97, 175, 239);
const META: Color = Color::Rgb(86, 182, 194);
const KEY: Color = Color::Rgb(224, 108, 117);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Token {
    Plain,
    Keyword,
    Type,
    String,
    Number,
    Comment,
    Function,
    /// Annotations, decorators, attributes, macros, preprocessor lines and
    /// shell variables.
    Meta,
    /// Object keys in JSON/YAML/TOML.
    Key,
}

pub fn style(token: Token) -> Style {
    let fg = |color| Style::default().fg(color);
    match token {
        Token::Plain => fg(PLAIN),
        Token::Keyword => fg(KEYWORD),
        Token::Type => fg(TYPE),
        Token::String => fg(STRING),
        Token::Number => fg(NUMBER),
        Token::Comment => fg(COMMENT).add_modifier(Modifier::ITALIC),
        Token::Function => fg(FUNCTION),
        Token::Meta => fg(META),
        Token::Key => fg(KEY),
    }
}

struct Lang {
    keywords: &'static [&'static str],
    /// Constants such as true/false/null, shown like numbers.
    literals: &'static [&'static str],
    types: &'static [&'static str],
    line_comments: &'static [&'static str],
    block_comment: Option<(&'static str, &'static str)>,
    quotes: &'static [char],
    triple_quotes: bool,
    /// Prefix of annotations/decorators (`@Override`, `@dataclass`).
    annotation: Option<char>,
    /// Rust `name!` macros and `#[attributes]`.
    rust_meta: bool,
    /// C preprocessor lines (`#include`).
    preprocessor: bool,
    /// `$VAR` shell variables.
    dollar_vars: bool,
    /// Capitalized identifiers are types (Java, Rust, C#...).
    capitalized_types: bool,
    /// Keys before `:` or `=` (JSON, YAML, TOML).
    keys: bool,
    /// HTML/XML tags.
    markup: bool,
    case_insensitive: bool,
}

const BASE: Lang = Lang {
    keywords: &[],
    literals: &["true", "false", "null"],
    types: &[],
    line_comments: &[],
    block_comment: None,
    quotes: &['"', '\''],
    triple_quotes: false,
    annotation: None,
    rust_meta: false,
    preprocessor: false,
    dollar_vars: false,
    capitalized_types: false,
    keys: false,
    markup: false,
    case_insensitive: false,
};

const C_LIKE: Lang = Lang {
    line_comments: &["//"],
    block_comment: Some(("/*", "*/")),
    capitalized_types: true,
    ..BASE
};

const RUST: Lang = Lang {
    keywords: &[
        "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else", "enum",
        "extern", "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut",
        "pub", "ref", "return", "static", "struct", "super", "trait", "type", "unsafe", "use",
        "where", "while",
    ],
    literals: &["true", "false", "self", "Self"],
    types: &[
        "i8", "i16", "i32", "i64", "i128", "isize", "u8", "u16", "u32", "u64", "u128", "usize",
        "f32", "f64", "bool", "char", "str",
    ],
    // Single quotes are lifetimes as often as char literals.
    quotes: &['"'],
    rust_meta: true,
    ..C_LIKE
};

const JAVA: Lang = Lang {
    keywords: &[
        "abstract", "assert", "break", "case", "catch", "class", "continue", "default", "do",
        "else", "enum", "extends", "final", "finally", "for", "if", "implements", "import",
        "instanceof", "interface", "native", "new", "package", "private", "protected", "public",
        "record", "return", "static", "super", "switch", "synchronized", "throw", "throws",
        "transient", "try", "var", "void", "volatile", "while", "yield",
    ],
    literals: &["true", "false", "null", "this"],
    types: &["boolean", "byte", "char", "double", "float", "int", "long", "short"],
    annotation: Some('@'),
    ..C_LIKE
};

const KOTLIN: Lang = Lang {
    keywords: &[
        "as", "break", "class", "companion", "continue", "data", "do", "else", "enum", "for",
        "fun", "if", "import", "in", "interface", "is", "object", "override", "package",
        "private", "protected", "public", "return", "sealed", "suspend", "throw", "try", "val",
        "var", "when", "while",
    ],
    literals: &["true", "false", "null", "this"],
    annotation: Some('@'),
    ..C_LIKE
};

const CSHARP: Lang = Lang {
    keywords: &[
        "abstract", "async", "await", "base", "break", "case", "catch", "class", "const",
        "continue", "default", "do", "else", "enum", "foreach", "for", "if", "in", "interface",
        "internal", "is", "namespace", "new", "out", "override", "private", "protected", "public",
        "readonly", "record", "ref", "return", "sealed", "static", "struct", "switch", "throw",
        "try", "using", "var", "virtual", "void", "while",
    ],
    literals: &["true", "false", "null", "this"],
    types: &[
        "bool", "byte", "char", "decimal", "double", "float", "int", "long", "object", "short",
        "string",
    ],
    ..C_LIKE
};

const C: Lang = Lang {
    keywords: &[
        "auto", "break", "case", "class", "const", "constexpr", "continue", "default", "delete",
        "do", "else", "enum", "extern", "for", "goto", "if", "inline", "namespace", "new",
        "private", "protected", "public", "return", "sizeof", "static", "struct", "switch",
        "template", "typedef", "typename", "union", "using", "virtual", "volatile", "while",
    ],
    literals: &["true", "false", "NULL", "nullptr", "this"],
    types: &[
        "bool", "char", "double", "float", "int", "long", "short", "signed", "size_t",
        "unsigned", "void",
    ],
    preprocessor: true,
    capitalized_types: false,
    ..C_LIKE
};

const GO: Lang = Lang {
    keywords: &[
        "break", "case", "chan", "const", "continue", "default", "defer", "else", "fallthrough",
        "for", "func", "go", "goto", "if", "import", "interface", "map", "package", "range",
        "return", "select", "struct", "switch", "type", "var",
    ],
    literals: &["true", "false", "nil", "iota"],
    types: &[
        "bool", "byte", "error", "float32", "float64", "int", "int8", "int16", "int32", "int64",
        "rune", "string", "uint", "uint8", "uint16", "uint32", "uint64", "any",
    ],
    quotes: &['"', '\'', '`'],
    capitalized_types: false,
    ..C_LIKE
};

const JAVASCRIPT: Lang = Lang {
    keywords: &[
        "async", "await", "break", "case", "catch", "class", "const", "continue", "default",
        "delete", "do", "else", "export", "extends", "finally", "for", "from", "function", "if",
        "import", "in", "instanceof", "interface", "let", "new", "of", "return", "static",
        "switch", "throw", "try", "type", "typeof", "var", "void", "while", "yield",
    ],
    literals: &["true", "false", "null", "undefined", "this", "NaN"],
    types: &["any", "boolean", "never", "number", "string", "unknown", "void"],
    quotes: &['"', '\'', '`'],
    annotation: Some('@'),
    ..C_LIKE
};

const PYTHON: Lang = Lang {
    keywords: &[
        "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del",
        "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is",
        "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with",
        "yield",
    ],
    literals: &["True", "False", "None", "self", "cls"],
    types: &[
        "bool", "bytes", "dict", "float", "int", "list", "object", "set", "str", "tuple",
    ],
    line_comments: &["#"],
    triple_quotes: true,
    annotation: Some('@'),
    capitalized_types: true,
    ..BASE
};

const SHELL: Lang = Lang {
    keywords: &[
        "case", "do", "done", "elif", "else", "esac", "export", "fi", "for", "function", "if",
        "in", "local", "return", "then", "until", "while",
    ],
    literals: &["true", "false"],
    line_comments: &["#"],
    dollar_vars: true,
    ..BASE
};

const POWERSHELL: Lang = Lang {
    keywords: &[
        "begin", "break", "catch", "continue", "do", "else", "elseif", "end", "finally",
        "foreach", "function", "if", "in", "param", "process", "return", "switch", "throw",
        "try", "until", "while",
    ],
    literals: &["$true", "$false", "$null"],
    line_comments: &["#"],
    block_comment: Some(("<#", "#>")),
    dollar_vars: true,
    case_insensitive: true,
    ..BASE
};

const SQL: Lang = Lang {
    keywords: &[
        "add", "alter", "and", "as", "asc", "by", "case", "create", "delete", "desc", "distinct",
        "drop", "else", "end", "exists", "from", "group", "having", "in", "index", "inner",
        "insert", "into", "is", "join", "key", "left", "like", "limit", "not", "on", "or",
        "order", "outer", "primary", "references", "right", "select", "set", "table", "then",
        "union", "update", "values", "view", "when", "where",
    ],
    types: &[
        "bigint", "boolean", "char", "date", "decimal", "float", "int", "integer", "text",
        "timestamp", "varchar",
    ],
    literals: &["true", "false", "null"],
    line_comments: &["--"],
    block_comment: Some(("/*", "*/")),
    quotes: &['\''],
    case_insensitive: true,
    ..BASE
};

const DATA: Lang = Lang {
    line_comments: &["#"],
    keys: true,
    ..BASE
};

const JSON: Lang = Lang {
    keys: true,
    quotes: &['"'],
    ..BASE
};

const MARKUP: Lang = Lang {
    literals: &[],
    block_comment: Some(("<!--", "-->")),
    quotes: &['"', '\''],
    markup: true,
    ..BASE
};

const GENERIC: Lang = Lang {
    line_comments: &["//", "#"],
    block_comment: Some(("/*", "*/")),
    quotes: &['"'],
    ..BASE
};

fn language(name: &str) -> &'static Lang {
    match name.trim().to_lowercase().as_str() {
        "rust" | "rs" => &RUST,
        "java" => &JAVA,
        "kotlin" | "kt" | "kts" => &KOTLIN,
        "c#" | "cs" | "csharp" => &CSHARP,
        "c" | "h" | "cpp" | "c++" | "cc" | "hpp" | "cxx" => &C,
        "go" | "golang" => &GO,
        "js" | "javascript" | "jsx" | "mjs" | "ts" | "typescript" | "tsx" => &JAVASCRIPT,
        "py" | "python" | "python3" => &PYTHON,
        "sh" | "bash" | "shell" | "zsh" | "console" => &SHELL,
        "ps1" | "powershell" | "pwsh" | "ps" => &POWERSHELL,
        "sql" => &SQL,
        "yaml" | "yml" | "toml" | "ini" | "properties" => &DATA,
        "json" | "jsonc" => &JSON,
        "html" | "xml" | "svg" | "xhtml" => &MARKUP,
        _ => &GENERIC,
    }
}

/// Highlights a code block line by line, carrying block comments and
/// multi-line strings over to the next line.
pub struct Highlighter {
    lang: &'static Lang,
    block_comment: bool,
    triple_quote: Option<&'static str>,
}

impl Highlighter {
    /// `info` is the text after the opening fence, e.g. "java".
    pub fn new(info: &str) -> Self {
        let name = info.split_whitespace().next().unwrap_or("");
        Self {
            lang: language(name),
            block_comment: false,
            triple_quote: None,
        }
    }

    pub fn line(&mut self, line: &str) -> Vec<Span<'static>> {
        self.tokens(line)
            .into_iter()
            .map(|(token, text)| Span::styled(text, style(token)))
            .collect()
    }

    pub fn tokens(&mut self, line: &str) -> Vec<(Token, String)> {
        let mut out = Tokens::default();
        let lang = self.lang;
        let mut rest = line;

        if lang.preprocessor && rest.trim_start().starts_with('#') {
            out.push(Token::Meta, rest);
            return out.finish();
        }

        while !rest.is_empty() {
            if self.block_comment {
                let end = lang.block_comment.map(|(_, end)| end).unwrap_or("");
                match rest.find(end) {
                    Some(at) => {
                        out.push(Token::Comment, &rest[..at + end.len()]);
                        rest = &rest[at + end.len()..];
                        self.block_comment = false;
                    }
                    None => {
                        out.push(Token::Comment, rest);
                        break;
                    }
                }
                continue;
            }
            if let Some(quote) = self.triple_quote {
                match rest.find(quote) {
                    Some(at) => {
                        out.push(Token::String, &rest[..at + 3]);
                        rest = &rest[at + 3..];
                        self.triple_quote = None;
                    }
                    None => {
                        out.push(Token::String, rest);
                        break;
                    }
                }
                continue;
            }

            if lang.line_comments.iter().any(|start| rest.starts_with(start)) {
                out.push(Token::Comment, rest);
                break;
            }
            if let Some((start, _)) = lang.block_comment.filter(|(start, _)| rest.starts_with(start)) {
                out.push(Token::Comment, start);
                rest = &rest[start.len()..];
                self.block_comment = true;
                continue;
            }
            if lang.triple_quotes {
                if let Some(quote) = ["\"\"\"", "'''"].into_iter().find(|q| rest.starts_with(q)) {
                    out.push(Token::String, quote);
                    rest = &rest[3..];
                    self.triple_quote = Some(quote);
                    continue;
                }
            }

            let ch = rest.chars().next().unwrap_or(' ');
            let used = if lang.quotes.contains(&ch) {
                let len = string_len(rest, ch);
                let after = rest[len..].trim_start();
                let token = if lang.keys && (after.starts_with(':') || after.starts_with('=')) {
                    Token::Key
                } else {
                    Token::String
                };
                out.push(token, &rest[..len]);
                len
            } else if ch.is_ascii_digit() && !out.ends_with_word() {
                let len = rest
                    .find(|c: char| !(c.is_ascii_alphanumeric() || c == '.' || c == '_'))
                    .unwrap_or(rest.len());
                out.push(Token::Number, &rest[..len]);
                len
            } else if lang.markup && ch == '<' {
                // "<tag", "</tag" or "<!DOCTYPE".
                let name_start = rest[1..]
                    .find(|c: char| c != '/' && c != '!' && c != '?')
                    .map_or(rest.len(), |at| at + 1);
                let name_len = word_len(&rest[name_start..]);
                out.push(Token::Plain, &rest[..name_start]);
                out.push(Token::Keyword, &rest[name_start..name_start + name_len]);
                name_start + name_len
            } else if lang.dollar_vars && ch == '$' {
                let len = 1 + rest[1..]
                    .find(|c: char| !(c.is_alphanumeric() || matches!(c, '_' | '{' | '}' | ':')))
                    .unwrap_or(rest.len() - 1);
                let word = &rest[..len];
                let token = if contains(lang.literals, word, true) { Token::Number } else { Token::Meta };
                out.push(token, word);
                len
            } else if lang.annotation == Some(ch) && rest[1..].starts_with(is_word_start) {
                let len = 1 + word_len(&rest[1..]);
                out.push(Token::Meta, &rest[..len]);
                len
            } else if lang.rust_meta && rest.starts_with("#[") || lang.rust_meta && rest.starts_with("#![") {
                let len = rest.find(']').map_or(rest.len(), |at| at + 1);
                out.push(Token::Meta, &rest[..len]);
                len
            } else if is_word_start(ch) {
                let len = word_len(rest);
                let word = &rest[..len];
                let after = &rest[len..];
                let token = self.classify(word, after);
                if token == Token::Meta && after.starts_with('!') {
                    out.push(token, &rest[..len + 1]);
                    len + 1
                } else {
                    out.push(token, word);
                    len
                }
            } else {
                out.push(Token::Plain, &rest[..ch.len_utf8()]);
                ch.len_utf8()
            };
            rest = &rest[used..];
        }
        out.finish()
    }

    fn classify(&self, word: &str, after: &str) -> Token {
        let lang = self.lang;
        let ci = lang.case_insensitive;
        let next = after.trim_start();
        if lang.keys && (next.starts_with(':') || next.starts_with('=')) {
            return Token::Key;
        }
        if contains(lang.keywords, word, ci) {
            return Token::Keyword;
        }
        if contains(lang.literals, word, ci) {
            return Token::Number;
        }
        if contains(lang.types, word, ci) {
            return Token::Type;
        }
        if lang.rust_meta && after.starts_with('!') && !after.starts_with("!=") {
            return Token::Meta;
        }
        if lang.markup && next.starts_with('=') {
            return Token::Key;
        }
        if next.starts_with('(') && !lang.markup {
            return Token::Function;
        }
        if lang.capitalized_types && word.starts_with(|c: char| c.is_ascii_uppercase()) {
            return Token::Type;
        }
        Token::Plain
    }
}

fn contains(list: &[&str], word: &str, case_insensitive: bool) -> bool {
    if case_insensitive {
        list.iter().any(|item| item.eq_ignore_ascii_case(word))
    } else {
        list.contains(&word)
    }
}

fn is_word_start(ch: char) -> bool {
    ch.is_alphabetic() || ch == '_'
}

fn word_len(text: &str) -> usize {
    text.find(|c: char| !(c.is_alphanumeric() || c == '_'))
        .unwrap_or(text.len())
}

/// Length of a string literal at the start of `text`, up to the closing
/// quote (respecting backslash escapes) or the end of the line.
fn string_len(text: &str, quote: char) -> usize {
    let mut escaped = false;
    for (at, ch) in text.char_indices().skip(1) {
        if escaped {
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == quote {
            return at + ch.len_utf8();
        }
    }
    text.len()
}

/// Collects tokens, merging neighbors of the same kind.
#[derive(Default)]
struct Tokens(Vec<(Token, String)>);

impl Tokens {
    fn push(&mut self, token: Token, text: &str) {
        if text.is_empty() {
            return;
        }
        match self.0.last_mut() {
            Some((last, buffer)) if *last == token => buffer.push_str(text),
            _ => self.0.push((token, text.to_string())),
        }
    }

    /// Whether the previous character belongs to a word, so digits in
    /// names like `utf8` are not numbers.
    fn ends_with_word(&self) -> bool {
        self.0
            .last()
            .and_then(|(_, text)| text.chars().last())
            .is_some_and(|c| c.is_alphanumeric() || c == '_')
    }

    fn finish(self) -> Vec<(Token, String)> {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(lang: &str, line: &str) -> Vec<(Token, String)> {
        Highlighter::new(lang).tokens(line)
    }

    /// Kind of the first token containing `text` (neighbors of the same
    /// kind are merged, e.g. " x: ").
    fn kind_of(tokens: &[(Token, String)], text: &str) -> Token {
        tokens
            .iter()
            .find(|(_, t)| t.contains(text))
            .unwrap_or_else(|| panic!("{text:?} not in {tokens:?}"))
            .0
    }

    #[test]
    fn java_line() {
        let t = tokens("java", r#"    public static void main(String[] args) { System.out.println("Hi"); } // done"#);
        assert_eq!(kind_of(&t, "public"), Token::Keyword);
        assert_eq!(kind_of(&t, "void"), Token::Keyword);
        assert_eq!(kind_of(&t, "main"), Token::Function);
        assert_eq!(kind_of(&t, "String"), Token::Type);
        assert_eq!(kind_of(&t, "println"), Token::Function);
        assert_eq!(kind_of(&t, "\"Hi\""), Token::String);
        assert_eq!(kind_of(&t, "// done"), Token::Comment);
        let text = t.iter().map(|(_, s)| s.as_str()).collect::<String>();
        assert!(text.ends_with("// done"));
    }

    #[test]
    fn rust_macros_attributes_and_numbers() {
        let t = tokens("rust", r#"#[derive(Debug)] let x: u8 = 42; println!("{x}"); utf8"#);
        assert_eq!(kind_of(&t, "#[derive(Debug)]"), Token::Meta);
        assert_eq!(kind_of(&t, "u8"), Token::Type);
        assert_eq!(kind_of(&t, "42"), Token::Number);
        assert_eq!(kind_of(&t, "println!"), Token::Meta);
        assert_eq!(kind_of(&t, "utf8"), Token::Plain);
    }

    #[test]
    fn python_triple_quotes_span_lines() {
        let mut highlighter = Highlighter::new("python");
        let first = highlighter.tokens(r#"doc = """start"#);
        assert_eq!(first.last().unwrap(), &(Token::String, "\"\"\"start".to_string()));
        let second = highlighter.tokens(r#"end""" + 1  # note"#);
        assert_eq!(kind_of(&second, "end\"\"\""), Token::String);
        assert_eq!(kind_of(&second, "1"), Token::Number);
        assert_eq!(kind_of(&second, "# note"), Token::Comment);
    }

    #[test]
    fn block_comments_span_lines() {
        let mut highlighter = Highlighter::new("c");
        assert_eq!(highlighter.tokens("int a; /* one")[2].0, Token::Comment);
        let next = highlighter.tokens("two */ return 0;");
        assert_eq!(kind_of(&next, "two */"), Token::Comment);
        assert_eq!(kind_of(&next, "return"), Token::Keyword);
        assert_eq!(highlighter.tokens("#include <stdio.h>")[0].0, Token::Meta);
    }

    #[test]
    fn data_keys_and_shell_variables() {
        let json = tokens("json", r#"{"name": "echo", "ok": true}"#);
        assert_eq!(kind_of(&json, "\"name\""), Token::Key);
        assert_eq!(kind_of(&json, "\"echo\""), Token::String);
        assert_eq!(kind_of(&json, "true"), Token::Number);

        let sh = tokens("bash", "echo \"$HOME\" $USER # hi");
        assert_eq!(kind_of(&sh, "$USER"), Token::Meta);
        assert_eq!(kind_of(&sh, "# hi"), Token::Comment);
    }

    #[test]
    fn sql_is_case_insensitive() {
        let t = tokens("sql", "SELECT name FROM users WHERE id = 1; -- all");
        assert_eq!(kind_of(&t, "SELECT"), Token::Keyword);
        assert_eq!(kind_of(&t, "-- all"), Token::Comment);
    }

    #[test]
    fn unknown_language_and_unicode_are_safe() {
        let t = tokens("", "café = \"ação\" // ok");
        let text = t.iter().map(|(_, s)| s.as_str()).collect::<String>();
        assert_eq!(text, "café = \"ação\" // ok");
        assert_eq!(kind_of(&t, "\"ação\""), Token::String);
    }
}
