//! A [`GraphSink`] that records what a Component's graph code built, for the
//! Component's host-independent tests.

use crate::graph::{GraphSink, KvRole, Node, Phase};

/// Records every [`GraphSink`] call, and answers with the edge ids the
/// Runtime's graph-builder gives, in a form a test can read:
///
/// | call | answer |
/// |---|---|
/// | `declare_input` | `in1`, `in2`... in the order asked |
/// | `weight_edge("w")` | `w:w`, the same for the same name |
/// | `kv_resource(2, K)` | `kv:2:K` |
/// | `add_node` of node `n` | `out:n` |
/// | `finish_graph("e")` | `graph(e)` |
#[derive(Debug, Default)]
pub struct RecordingSink {
    /// `(phase, sequence length, position offset)` of each `begin_graph`.
    pub began: Vec<(Phase, u64, u64)>,
    /// `(name, shape)` of each declared input.
    pub inputs: Vec<(String, Vec<u64>)>,
    /// The name of each `weight_edge` call, repeats included.
    pub weights: Vec<String>,
    /// `(edge, target)` of each alias.
    pub aliases: Vec<(String, String)>,
    /// Every node added, in order.
    pub nodes: Vec<Node>,
    inputs_declared: u32,
}

impl RecordingSink {
    /// A sink that has recorded nothing.
    pub fn new() -> Self {
        Self::default()
    }

    /// The node with id `id`.
    ///
    /// # Panics
    ///
    /// When no node has that id: a test that asks for a node the graph does
    /// not hold has found its failure.
    pub fn node(&self, id: &str) -> &Node {
        self.nodes
            .iter()
            .find(|node| node.id == id)
            .unwrap_or_else(|| panic!("no node {id}"))
    }

    /// Whether a node has id `id`.
    pub fn has(&self, id: &str) -> bool {
        self.nodes.iter().any(|node| node.id == id)
    }
}

impl GraphSink for RecordingSink {
    fn begin_graph(&mut self, phase: Phase, sequence_length: u64, position_offset: u64) {
        self.began.push((phase, sequence_length, position_offset));
    }

    fn declare_input(&mut self, name: &str, shape: &[u64]) -> String {
        self.inputs.push((name.to_string(), shape.to_vec()));
        self.inputs_declared += 1;
        format!("in{}", self.inputs_declared)
    }

    fn weight_edge(&mut self, name: &str) -> String {
        self.weights.push(name.to_string());
        format!("w:{name}")
    }

    fn alias_weight_edge(&mut self, edge: &str, target: &str) {
        self.aliases.push((edge.to_string(), target.to_string()));
    }

    fn kv_resource(&mut self, layer: u64, role: KvRole) -> String {
        format!("kv:{layer}:{role:?}")
    }

    fn add_node(&mut self, node: Node) -> String {
        let output = format!("out:{}", node.id);
        self.nodes.push(node);
        output
    }

    fn finish_graph(&mut self, output: &str) -> String {
        format!("graph({output})")
    }
}

#[cfg(test)]
mod tests;
