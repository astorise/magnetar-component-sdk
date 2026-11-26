//! The graph a Component describes, and the sink it describes it to.
//!
//! These are the host-independent mirror of the Runtime's `graph-builder`
//! Capability: a Component's graph code is written against [`GraphSink`], so
//! that the same code runs against the real Capability (through
//! [`wit_graph_sink!`](crate::wit_graph_sink)) and against a recording double
//! in tests ([`RecordingSink`](crate::testing::RecordingSink)).

/// Which graph a Component is asked to build.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase {
    /// All the prompt's rows at once.
    Prefill,
    /// One new token against the cached history.
    Decode,
}

/// Which half of a layer's attention history a KV resource holds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvRole {
    /// Keys.
    K,
    /// Values.
    V,
}

/// A node attribute's value: the portable subset the Runtime's graph-builder
/// takes (`Boolean`, `Integer`, `Float` and `Text`).
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Self::Int(value)
    }
}

impl From<u32> for Value {
    fn from(value: u32) -> Self {
        Self::Int(i64::from(value))
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}

impl From<&str> for Value {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<String> for Value {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

/// One named attribute of a node.
#[derive(Clone, Debug, PartialEq)]
pub struct Attr {
    pub name: &'static str,
    pub value: Value,
}

impl Attr {
    /// An attribute `name` with `value`.
    pub fn new(name: &'static str, value: impl Into<Value>) -> Self {
        Self {
            name,
            value: value.into(),
        }
    }
}

/// A node's involvement with a KV resource: it reads the history, or appends
/// to it.
#[derive(Clone, Debug, PartialEq)]
pub struct Kv {
    /// The resource id [`GraphSink::kv_resource`] returned.
    pub resource: String,
    pub role: KvRole,
    /// Whether the node appends its output rows to the history.
    pub append: bool,
}

/// One operator node of the graph.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    /// The node's id, unique in its graph.
    pub id: String,
    /// The operator's name in the Runtime's catalog (`"matmul"`, `"rope"`...).
    pub operator: &'static str,
    /// The operator's family (`"LinearAlgebra"`, `"Tensor"`...).
    pub family: &'static str,
    pub attrs: Vec<Attr>,
    /// The edge ids the node reads, in the operator's input order.
    pub inputs: Vec<String>,
    /// The shape of the node's one output.
    pub shape: Vec<u64>,
    pub kv: Option<Kv>,
}

/// What a Component's graph code writes to: the `graph-builder` Capability,
/// as far as a graph with one output per node needs it.
///
/// Edge ids are opaque strings the sink hands back and the Component passes on
/// as the `inputs` of later nodes.
pub trait GraphSink {
    /// Starts a graph of `phase` for `sequence_length` rows, the first at
    /// absolute position `position_offset`.
    fn begin_graph(&mut self, phase: Phase, sequence_length: u64, position_offset: u64);
    /// Declares a graph input of `shape`, and returns its edge.
    fn declare_input(&mut self, name: &str, shape: &[u64]) -> String;
    /// The edge of the weight the Runtime knows as `name` (its canonical
    /// tensor name). Asking again for the same name gives the same edge.
    fn weight_edge(&mut self, name: &str) -> String;
    /// Makes `edge` read the weight named `target` instead (a tied head reads
    /// the token embedding).
    fn alias_weight_edge(&mut self, edge: &str, target: &str);
    /// The KV resource of `layer`'s `role`.
    fn kv_resource(&mut self, layer: u64, role: KvRole) -> String;
    /// Adds `node` and returns its output edge.
    fn add_node(&mut self, node: Node) -> String;
    /// Ends the graph at `output` and returns its handle.
    fn finish_graph(&mut self, output: &str) -> String;
}

#[cfg(test)]
mod tests;
