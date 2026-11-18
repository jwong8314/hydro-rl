//! Deploy the coordinator as a localhost process that Python clients reach over
//! TCP with newline-delimited JSON (see `python/hydro_rl/client.py`).
//!
//! cargo run --example serve -- [port] [max_staleness|none] [group_size] [oldest_first]
//!
//! Defaults match Miles' `DefaultDataBuffer`: no staleness limit (Miles'
//! `--max-weight-staleness` defaults to None), no group-size check (0), FIFO.
//! The Miles adapter checks these against its own launch arguments at connect.

use hydro_deploy::Deployment;
use hydro_lang::location::{Location, NetworkHint};
use tokio_util::codec::LinesCodec;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let port: u16 = args.get(1).map_or(7878, |s| s.parse().unwrap());
    let max_staleness: u64 = match args.get(2).map(String::as_str) {
        None | Some("none") => rl_coordinator::protocol::NO_STALENESS_LIMIT,
        Some(s) => s.parse().unwrap(),
    };
    let group_size: u32 = args.get(3).map_or(0, |s| s.parse().unwrap());
    let oldest_first: bool = args.get(4).is_some_and(|s| s.parse().unwrap());

    let mut deployment = Deployment::new();
    let mut flow = hydro_lang::compile::builder::FlowBuilder::new();
    let process = flow.process::<rl_coordinator::Coord>();
    let external = flow.external::<()>();

    let (_port, lines, _membership, replies) = process
        .bidi_external_many_bytes::<_, String, LinesCodec>(
            &external,
            NetworkHint::TcpPort(Some(port)),
        );
    replies.complete(rl_coordinator::coordinator_json(
        lines,
        max_staleness,
        group_size,
        3,    // starvation_threshold
        oldest_first,
    ));

    let _nodes = flow
        .with_process(&process, deployment.Localhost())
        .with_external(&external, deployment.Localhost())
        .deploy(&mut deployment);

    deployment.deploy().await.unwrap();
    println!(
        "coordinator listening on 127.0.0.1:{port} (max_staleness={}, group_size={group_size}, oldest_first={oldest_first})",
        if max_staleness == rl_coordinator::protocol::NO_STALENESS_LIMIT { "none".to_string() } else { max_staleness.to_string() }
    );
    deployment.start_ctrl_c().await.unwrap();
}
