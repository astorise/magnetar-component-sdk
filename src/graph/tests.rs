//! Unit tests for the parent module.

use super::*;

/// An attribute takes any value a node can carry, from the Rust type it
/// already has.
#[test]
fn an_attribute_is_made_from_the_value_it_holds() {
    assert_eq!(Attr::new("causal", true).value, Value::Bool(true));
    assert_eq!(Attr::new("heads", 32u32).value, Value::Int(32));
    assert_eq!(Attr::new("position", -3i64).value, Value::Int(-3));
    assert_eq!(Attr::new("epsilon", 1e-6).value, Value::Float(1e-6));
    assert_eq!(Attr::new("kind", "silu").value, Value::Text("silu".into()));
    assert_eq!(
        Attr::new("name", String::from("q")).value,
        Value::Text("q".into())
    );
    // The value itself is accepted unchanged.
    assert_eq!(Attr::new("raw", Value::Int(7)).value, Value::Int(7));
    assert_eq!(Attr::new("raw", true).name, "raw");
}
