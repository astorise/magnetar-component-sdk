# magnetar-component-sdk

## Purpose

Shared building blocks for the Model Components of the
[Magnetar](https://github.com/astorise/Magnetar) local AI Runtime.

A Model Component describes a model's prefill and decode graphs to the Runtime
through its `graph-builder` Capability. Before this crate, every Component
carried its own copy of the same scaffolding for that, and the copies drifted
(the adapter of `gemma` differed from `mistral`'s and `olmo`'s only in argument
names, which is how copies drift). A new model family should be a new
Component and nothing else; this crate is where what Components share lives, so
that a Component holds only what is specific to its family.

## What it holds

| Module | What |
|---|---|
| `graph` | The graph description (`Node`, `Attr`, `Value`, `Kv`, `Phase`, `KvRole`) and `GraphSink`, the trait a Component's graph code is written against, so that the same code runs against the real Capability and against a test double. |
| `wit_graph_sink!` | Implements `GraphSink` for a type on the `graph-builder` interface of the bindings a Component generated with `wit-bindgen`. |
| `testing` | `RecordingSink`, a `GraphSink` that records every call, for a Component's host-independent tests. |

```rust
// component.rs: the adapter to the Runtime, written once.
wit_bindgen::generate!({
    path: "wit/model-component-graph.wit",
    world: "model-component-graph-producer",
});
use magnetar::model_component_graph::graph_builder as gb;

struct Host;
magnetar_component_sdk::wit_graph_sink!(Host, gb);

// plan.rs: the graph, host-independent, tested with `RecordingSink`.
pub use magnetar_component_sdk::graph::{Attr, GraphSink, Kv, KvRole, Node, Phase, Value};
```

## What does not belong here

- Anything that names a model family.
- Anything that depends on the Runtime (`magnetar-runtime`): this crate has no
  dependency at all, so that a Component built for `wasm32-wasip2` pulls in
  nothing it does not use.
- The generated WIT bindings: each Component generates its own, against the
  contract version it targets. The contract is additive, so a Component built
  against 1.8.0 keeps running on a Runtime at a later version; the macro relies
  only on the `graph-builder` items that exist since 1.8.0.

## Status

Initial release: the graph description, the sink, the WIT adapter and the
recording double, extracted from the Mistral, OLMo and Gemma 4 Components, whose
copies they replace. Checked against the real bindings of the Mistral
Component. The shared decoder blocks (normalization, rotary embedding,
attention, gated MLP, mixture of experts) and the configuration helpers follow,
Component by Component, each migration keeping the graph it emits byte for byte.

## Using it

While the API settles, a Component names the crate as a git dependency pinned
by commit, and its committed `Cargo.lock` records it:

```toml
[dependencies]
magnetar-component-sdk = { git = "https://github.com/astorise/magnetar-component-sdk", rev = "<commit>" }
```

Never a branch or a tag: they move, and the Component's build would move with
them. Taking a newer commit is a pull request of the Component's own. Once the
API has been through the Components and is stable, the crate is published to
crates.io and Components depend on it by version. Changes to the crate stay
additive (a default on a new method), so a Component on an older commit keeps
building.

## Checks

```sh
cargo fmt -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps
```
