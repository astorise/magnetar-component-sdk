//! Shared building blocks for Magnetar Model Components.
//!
//! A Model Component describes a model's prefill and decode graphs to the
//! Runtime through its `graph-builder` Capability; it never selects a
//! Provider or Device and never sees a native handle. Every Component used
//! to carry its own copy of the same scaffolding for that: the types that
//! describe a node, the trait the graph code writes to (so that it can be
//! tested without a WebAssembly host), the adapter from that trait to the
//! generated WIT bindings, and a test double that records what was built.
//! This crate holds them once.
//!
//! - [`graph`]: the graph description ([`graph::Node`], [`graph::Attr`],
//!   [`graph::Kv`]...) and [`graph::GraphSink`], the trait a Component's graph
//!   code is written against.
//! - [`wit_graph_sink!`]: implements [`graph::GraphSink`] on a type for the
//!   `graph-builder` interface of the bindings a Component generated with
//!   `wit-bindgen`.
//! - [`testing`]: [`testing::RecordingSink`], a [`graph::GraphSink`] that
//!   records every call, for a Component's host-independent tests.
//!
//! What does not belong here: anything that names a model family, anything
//! that depends on the Runtime, and the generated bindings themselves (each
//! Component generates its own, against the contract version it targets).

pub mod graph;
pub mod testing;
mod wit;
