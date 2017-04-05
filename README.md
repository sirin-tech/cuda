# Cuda

NVIDIA GPU CUDA library bindings for Erlang and Elixir.

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `gpu_math` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [{:cuda, "~> 0.1.0"}]
end
```

## Debugging

To have some debug messages from C++ cuda binding compile library with
`GPU_DEBUG=1` environment variable like this:

```sh
mix clean
GPU_DEBUG=1 mix compile
```
