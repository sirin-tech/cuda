# Cuda

NVIDIA GPU CUDA library bindings for Erlang and Elixir.

## Installation

```elixir
def deps do
  [{:cuda, "~> 0.1.0"}]
end
```

## Prerequisite

At least one of video cards should be not in exclusive or prohibited compute
mode. To check your video card mode run:

```sh
nvidia-smi --format=csv --query-gpu="compute_mode"
```

To change comute mode to default run:

```sh
sudo nvidia-smi -c 0
```

## Debugging

To have some debug messages from C++ cuda binding, compile library with
`GPU_DEBUG=1` environment variable like this:

```sh
mix clean
GPU_DEBUG=1 mix compile
```
