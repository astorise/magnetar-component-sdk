//! The adapter from [`GraphSink`](crate::graph::GraphSink) to the `graph-builder`
//! interface of a Component's generated WIT bindings.

/// Implements [`GraphSink`](crate::graph::GraphSink) for `$host` by calling
/// the `graph-builder` interface of the bindings a Component generated.
///
/// `$gb` is the name, in scope where the macro is called, of that interface's
/// module: `magnetar::model_component_graph::graph_builder` as `gb`.
///
/// ```ignore
/// wit_bindgen::generate!({
///     path: "wit/model-component-graph.wit",
///     world: "model-component-graph-producer",
/// });
///
/// use magnetar::model_component_graph::graph_builder as gb;
///
/// struct Host;
/// magnetar_component_sdk::wit_graph_sink!(Host, gb);
/// ```
///
/// The macro needs, from `$gb`, the types and functions the contract's
/// `graph-builder` has had since 1.8.0: `GraphPhase`, `KvCacheRole`,
/// `TensorShape { dimensions }`, `Attribute { name, value }` with
/// `AttributeValue::{Boolean, Integer, FloatValue, Text}`,
/// `KvInvolvement { resource_id, role, append }`, and `begin_graph`,
/// `declare_input`, `weight_edge`, `alias_weight_edge`, `kv_resource`,
/// `add_node` and `finish_graph`. A contract version that changes one of them
/// fails to compile at the call, which is the point: the adapter was written
/// once, and a change is made once.
#[macro_export]
macro_rules! wit_graph_sink {
    ($host:ty, $gb:ident) => {
        impl $crate::graph::GraphSink for $host {
            fn begin_graph(
                &mut self,
                phase: $crate::graph::Phase,
                sequence_length: u64,
                position_offset: u64,
            ) {
                let phase = match phase {
                    $crate::graph::Phase::Prefill => $gb::GraphPhase::Prefill,
                    $crate::graph::Phase::Decode => $gb::GraphPhase::Decode,
                };
                $gb::begin_graph(phase, sequence_length, position_offset);
            }

            fn declare_input(&mut self, name: &str, shape: &[u64]) -> ::std::string::String {
                $gb::declare_input(
                    name,
                    &$gb::TensorShape {
                        dimensions: shape.to_vec(),
                    },
                )
            }

            fn weight_edge(&mut self, name: &str) -> ::std::string::String {
                $gb::weight_edge(name)
            }

            fn alias_weight_edge(&mut self, edge: &str, target: &str) {
                $gb::alias_weight_edge(edge, target);
            }

            fn kv_resource(
                &mut self,
                layer: u64,
                role: $crate::graph::KvRole,
            ) -> ::std::string::String {
                $gb::kv_resource(
                    layer,
                    match role {
                        $crate::graph::KvRole::K => $gb::KvCacheRole::K,
                        $crate::graph::KvRole::V => $gb::KvCacheRole::V,
                    },
                )
            }

            fn add_node(&mut self, node: $crate::graph::Node) -> ::std::string::String {
                let attrs: ::std::vec::Vec<$gb::Attribute> = node
                    .attrs
                    .iter()
                    .map(|attr| $gb::Attribute {
                        name: attr.name.to_string(),
                        value: match &attr.value {
                            $crate::graph::Value::Bool(value) => {
                                $gb::AttributeValue::Boolean(*value)
                            }
                            $crate::graph::Value::Int(value) => {
                                $gb::AttributeValue::Integer(*value)
                            }
                            $crate::graph::Value::Float(value) => {
                                $gb::AttributeValue::FloatValue(*value)
                            }
                            $crate::graph::Value::Text(value) => {
                                $gb::AttributeValue::Text(value.clone())
                            }
                        },
                    })
                    .collect();
                let kv = node.kv.as_ref().map(|kv| $gb::KvInvolvement {
                    resource_id: kv.resource.clone(),
                    role: match kv.role {
                        $crate::graph::KvRole::K => $gb::KvCacheRole::K,
                        $crate::graph::KvRole::V => $gb::KvCacheRole::V,
                    },
                    append: kv.append,
                });
                $gb::add_node(
                    &node.id,
                    node.operator,
                    node.family,
                    &attrs,
                    &node.inputs,
                    &$gb::TensorShape {
                        dimensions: node.shape,
                    },
                    kv.as_ref(),
                )
            }

            fn finish_graph(&mut self, output: &str) -> ::std::string::String {
                $gb::finish_graph(output)
            }
        }
    };
}

#[cfg(test)]
mod tests;
