defmodule GPUMath do
  @moduledoc """
  NVIDIA GPU CUDA library bindings for Erlang and Elixir.
  """

  use GenServer

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Returns NVIDIA driver and CUDA library info

  param can be ommitted or must be one of:

  * device_count - get GPU device count
  * driver_version - get CUDA driver version
  * memory - get free and total GPU memory
  * runtime_version - get CUDA runtime version
  """
  @spec info(nil | :device_count | :driver_version | :memory | :runtime_version) :: any
  def info(info \\ nil) do
    GenServer.call(__MODULE__, {:call, :info, info})
  end

  def init(opts) do
    cmd = Keyword.get(opts, :port_bin, "priv/gpu_math_port")
    port = Port.open({:spawn, cmd}, [:binary, :nouse_stdio, packet: 4])
    {:ok, port}
  end

  def handle_call({:call, func, arg}, _from, port) do
    Port.command(port, :erlang.term_to_binary({func, arg}))
    receive do
      {^port, {:data, data}} -> {:reply, :erlang.binary_to_term(data), port}
      _ -> {:reply, {:error, "Port communication error"}, port}
    end
  end
end
