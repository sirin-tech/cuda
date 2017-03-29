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

  def init_driver(device \\ nil) do
    GenServer.call(__MODULE__, {:call, :init, device})
  end

  def compile(sources, opts \\ nil)
  def compile(source, opts) when is_binary(source) do
    GenServer.call(__MODULE__, {:call, :compile, {[source], opts}})
  end
  def compile(sources, opts) when is_list(sources) do
    GenServer.call(__MODULE__, {:call, :compile, {sources, opts}})
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

  def test_ptx do
    ".version 3.2\n" <>
    ".target sm_20\n" <>
    ".address_size 64\n\n" <>
    ".visible .entry _Z8myKernelPi(\n" <>
	  "\t.param .u64 _Z8myKernelPi_param_0\n" <>
    ")\n" <>
    "{\n" <>
	  "\t.reg .s32 	%r<5>;\n" <>
	  "\t.reg .s64 	%rd<5>;\n\n" <>
  	"\tld.param.u64 	%rd1, [_Z8myKernelPi_param_0];\n" <>
  	"\tcvta.to.global.u64 	%rd2, %rd1;\n" <>
  	"\t.loc 1 3 1\n" <>
  	"\tmov.u32 	%r1, %ntid.x;\n" <>
  	"\tmov.u32 	%r2, %ctaid.x;\n" <>
  	"\tmov.u32 	%r3, %tid.x;\n" <>
  	"\tmad.lo.s32 	%r4, %r1, %r2, %r3;\n" <>
  	"\tmul.wide.s32 	%rd3, %r4, 4;\n" <>
  	"\tadd.s64 	%rd4, %rd2, %rd3;\n" <>
  	"\t.loc 1 4 1\n" <>
  	"\tst.global.u32 	[%rd4], %r4;\n" <>
  	"\t.loc 1 5 2\n" <>
  	"\tret;\n" <>
    "}\n"
  end
end
