//! Unit tests for the parent module.
//!
//! `gb` stands in for the module `wit-bindgen` generates for the
//! `graph-builder` interface, with the types and signatures the macro relies
//! on; each call it receives is written down as text.

use crate::graph::{Attr, GraphSink, Kv, KvRole, Node, Phase};
mod gb {
    use std::cell::RefCell;

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub enum GraphPhase {
        Prefill,
        Decode,
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub enum KvCacheRole {
        K,
        V,
    }

    #[derive(Clone, Debug, PartialEq)]
    pub struct TensorShape {
        pub dimensions: Vec<u64>,
    }

    #[derive(Clone, Debug, PartialEq)]
    pub enum AttributeValue {
        Boolean(bool),
        Integer(i64),
        FloatValue(f64),
        Text(String),
    }

    #[derive(Clone, Debug, PartialEq)]
    pub struct Attribute {
        pub name: String,
        pub value: AttributeValue,
    }

    #[derive(Clone, Debug, PartialEq)]
    pub struct KvInvolvement {
        pub resource_id: String,
        pub role: KvCacheRole,
        pub append: bool,
    }

    thread_local! {
        pub static CALLS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    }

    fn note(call: String) {
        CALLS.with(|calls| calls.borrow_mut().push(call));
    }

    pub fn begin_graph(phase: GraphPhase, sequence_length: u64, position_offset: u64) {
        note(format!(
            "begin {phase:?} {sequence_length} {position_offset}"
        ));
    }

    pub fn declare_input(name: &str, shape: &TensorShape) -> String {
        note(format!("input {name} {:?}", shape.dimensions));
        format!("input:{name}")
    }

    pub fn weight_edge(name: &str) -> String {
        note(format!("weight {name}"));
        format!("weight:{name}")
    }

    pub fn alias_weight_edge(edge: &str, target: &str) {
        note(format!("alias {edge} {target}"));
    }

    pub fn kv_resource(layer: u64, role: KvCacheRole) -> String {
        note(format!("kv {layer} {role:?}"));
        format!("kv:{layer}:{role:?}")
    }

    pub fn add_node(
        id: &str,
        operator: &str,
        family: &str,
        attributes: &[Attribute],
        inputs: &[String],
        shape: &TensorShape,
        kv: Option<&KvInvolvement>,
    ) -> String {
        note(format!(
            "node {id} {operator} {family} {attributes:?} {inputs:?} {:?} {kv:?}",
            shape.dimensions
        ));
        format!("edge:{id}")
    }

    pub fn finish_graph(output: &str) -> String {
        note(format!("finish {output}"));
        format!("graph:{output}")
    }
}

struct Host;

crate::wit_graph_sink!(Host, gb);

fn calls() -> Vec<String> {
    gb::CALLS.with(|calls| std::mem::take(&mut *calls.borrow_mut()))
}

/// Every call reaches the interface with its arguments translated, and its
/// answer comes back unchanged.
#[test]
fn every_call_reaches_the_interface_and_its_answer_comes_back() {
    calls();
    let mut host = Host;
    host.begin_graph(Phase::Prefill, 7, 0);
    host.begin_graph(Phase::Decode, 1, 7);
    assert_eq!(
        host.declare_input("input.token_ids", &[7]),
        "input:input.token_ids"
    );
    assert_eq!(host.weight_edge("lm_head"), "weight:lm_head");
    host.alias_weight_edge("weight:lm_head", "token_embedding");
    assert_eq!(host.kv_resource(3, KvRole::K), "kv:3:K");
    assert_eq!(host.kv_resource(3, KvRole::V), "kv:3:V");
    assert_eq!(host.finish_graph("edge:x"), "graph:edge:x");
    assert_eq!(
        calls(),
        [
            "begin Prefill 7 0",
            "begin Decode 1 7",
            "input input.token_ids [7]",
            "weight lm_head",
            "alias weight:lm_head token_embedding",
            "kv 3 K",
            "kv 3 V",
            "finish edge:x",
        ]
    );
}

/// A node reaches the interface with every attribute kind, its inputs, its
/// output shape and its KV involvement translated.
#[test]
fn a_node_reaches_the_interface_with_everything_it_carries() {
    calls();
    let mut host = Host;
    let output = host.add_node(Node {
        id: "layer0.attention".into(),
        operator: "attention",
        family: "LinearAlgebra",
        attrs: vec![
            Attr::new("causal", true),
            Attr::new("heads", 8u32),
            Attr::new("scale", 0.25),
            Attr::new("kind", "gqa"),
        ],
        inputs: vec!["edge:q".into(), "edge:k".into(), "edge:v".into()],
        shape: vec![5, 64],
        kv: Some(Kv {
            resource: "kv:0:V".into(),
            role: KvRole::V,
            append: true,
        }),
    });
    assert_eq!(output, "edge:layer0.attention");
    assert_eq!(
        calls(),
        [concat!(
            "node layer0.attention attention LinearAlgebra ",
            "[Attribute { name: \"causal\", value: Boolean(true) }, ",
            "Attribute { name: \"heads\", value: Integer(8) }, ",
            "Attribute { name: \"scale\", value: FloatValue(0.25) }, ",
            "Attribute { name: \"kind\", value: Text(\"gqa\") }] ",
            "[\"edge:q\", \"edge:k\", \"edge:v\"] [5, 64] ",
            "Some(KvInvolvement { resource_id: \"kv:0:V\", role: V, append: true })"
        )]
    );
}

/// A node with no KV involvement and no attribute carries none.
#[test]
fn a_bare_node_carries_nothing_extra() {
    calls();
    let mut host = Host;
    host.add_node(Node {
        id: "n".into(),
        operator: "add",
        family: "Tensor",
        attrs: vec![],
        inputs: vec!["a".into(), "b".into()],
        shape: vec![3],
        kv: None,
    });
    assert_eq!(calls(), ["node n add Tensor [] [\"a\", \"b\"] [3] None"]);
}
