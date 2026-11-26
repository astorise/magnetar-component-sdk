//! Unit tests for the parent module.

use super::*;
use crate::graph::{Attr, Kv};

fn node(id: &str) -> Node {
    Node {
        id: id.into(),
        operator: "matmul",
        family: "LinearAlgebra",
        attrs: vec![Attr::new("transpose_b", false)],
        inputs: vec!["in1".into(), "w:layers.0.q_proj".into()],
        shape: vec![4, 8],
        kv: Some(Kv {
            resource: "kv:0:K".into(),
            role: KvRole::K,
            append: true,
        }),
    }
}

/// A build is recorded in the order it was made, and every answer has the
/// documented form.
#[test]
fn a_build_is_recorded_and_answered_in_the_documented_form() {
    let mut sink = RecordingSink::new();
    sink.begin_graph(Phase::Prefill, 4, 0);
    assert_eq!(sink.declare_input("input.token_ids", &[4]), "in1");
    assert_eq!(sink.declare_input("input.positions", &[4]), "in2");
    assert_eq!(sink.weight_edge("lm_head"), "w:lm_head");
    sink.alias_weight_edge("w:lm_head", "token_embedding");
    assert_eq!(sink.kv_resource(2, KvRole::V), "kv:2:V");
    assert_eq!(sink.add_node(node("layer0.q")), "out:layer0.q");
    assert_eq!(sink.finish_graph("out:layer0.q"), "graph(out:layer0.q)");

    assert_eq!(sink.began, [(Phase::Prefill, 4, 0)]);
    assert_eq!(
        sink.inputs,
        [
            ("input.token_ids".to_string(), vec![4]),
            ("input.positions".to_string(), vec![4])
        ]
    );
    assert_eq!(sink.weights, ["lm_head"]);
    assert_eq!(
        sink.aliases,
        [("w:lm_head".to_string(), "token_embedding".to_string())]
    );
    assert_eq!(sink.nodes.len(), 1);
}

/// A weight asked for twice has one edge, as in the Runtime, and both asks
/// are recorded.
#[test]
fn a_weight_asked_for_twice_has_one_edge() {
    let mut sink = RecordingSink::new();
    assert_eq!(sink.weight_edge("w"), sink.weight_edge("w"));
    assert_eq!(sink.weights, ["w", "w"]);
}

/// A node is found by id, and a missing one is a failure, not a default.
#[test]
fn a_node_is_found_by_its_id() {
    let mut sink = RecordingSink::new();
    sink.add_node(node("a"));
    sink.add_node(node("b"));
    assert!(sink.has("a") && sink.has("b") && !sink.has("c"));
    assert_eq!(sink.node("b").id, "b");
    assert_eq!(sink.node("a").shape, [4, 8]);
}

#[test]
#[should_panic(expected = "no node missing")]
fn asking_for_a_node_the_graph_does_not_hold_fails() {
    RecordingSink::new().node("missing");
}
