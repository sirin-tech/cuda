defmodule Cuda do
  @moduledoc """
  NVIDIA GPU CUDA library bindings for Erlang and Elixir.
  """

  use GenServer
  require Logger

  @term_call <<1>>
  @raw_call  <<2>>

  # defdelegate start_driver(opts \\ []), to: Cuda.App

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, [])
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
  def info(pid, info \\ nil) do
    GenServer.call(pid, {:call, :info, info})
  end

  def compile(pid, sources, opts \\ nil)
  def compile(pid, {:file, file}, opts) when is_binary(file) do
    with {:ok, source} <- File.read(file) do
      GenServer.call(pid, {:call, :compile, {[source], opts}})
    end
  end
  def compile(pid, {:files, files}, opts) when is_list(files) do
    sources = Enum.reduce(files, {:ok, []}, fn
      {:ok, sources}, file ->
        with {:ok, source} <- File.read(file), do: [source | sources]
      error, _ ->
        error
    end)
    with {:ok, sources} <- sources do
      GenServer.call(pid, {:call, :compile, {sources, opts}})
    end
  end
  def compile(pid, source, opts) when is_binary(source) do
    GenServer.call(pid, {:call, :compile, {[source], opts}})
  end
  def compile(pid, sources, opts) when is_list(sources) do
    GenServer.call(pid, {:call, :compile, {sources, opts}})
  end

  def module_load(pid, src, opts \\ []) do
    GenServer.call(pid, {:call, :module_load, {src, opts}})
  end

  def memory_load(pid, data) do
    GenServer.call(pid, {:call_raw, :memory_load, data})
  end

  def memory_read(pid, handle) do
    GenServer.call(pid, {:call, :memory_read, handle})
  end

  def memory_unload(pid, handle) do
    GenServer.call(pid, {:call, :memory_unload, handle})
  end

  def run(pid, module, func) do
    GenServer.call(pid, {:call, :run, {module, func}})
  end
  def run(pid, module, func, params) do
    GenServer.call(pid, {:call, :run, {module, func, params}})
  end
  def run(pid, module, func, block, params) do
    GenServer.call(pid, {:call, :run, {module, func, block, params}})
  end
  def run(pid, module, func, block, grid, params) do
    GenServer.call(pid, {:call, :run, {module, func, block, grid, params}})
  end

  def init(opts) do
    #Process.flag(:trap_exit, true)
    # {device, opts} = Keyword.pop(opts, :device)
    cmd = Keyword.get(opts, :port_bin, Application.app_dir(:cuda, Path.join(~w(priv cuda_port))))
    cmd = case Keyword.get(opts, :device) do
      nil    -> cmd
      device -> "#{cmd} #{device}"
    end
    port = Port.open({:spawn, cmd}, [:binary, :nouse_stdio, packet: 4])
    {:ok, port}
  end

  #def terminate(_, port) do
  #  IO.inspect("TERMINATE")
  #  unless is_nil(Port.info(port)) do
  #    Port.command(port, @term_call <> :erlang.term_to_binary({:exit, nil}))
  #    # Give the port a chance to gracefully exit
  #    Process.sleep(50)
  #    IO.inspect("TERMINATING PORT")
  #    Port.close(port)
  #  end
  #  :ok
  #end

  def handle_call({:call, func, arg}, _from, port) do
    Port.command(port, @term_call <> :erlang.term_to_binary({func, arg}))
    wait_reply(port)
  end

  def handle_call({:call_raw, func, arg}, _from, port) do
    Port.command(port, @raw_call <> raw_func(func) <> arg)
    wait_reply(port)
  end

  defp raw_func(:memory_load), do: <<1>>

  defp wait_reply(port) do
    receive do
      {^port, {:data, @term_call <> data}} -> {:reply, :erlang.binary_to_term(data), port}
      {^port, {:data, @raw_call <> data}} -> {:reply, {:ok, data}, port}
      _ -> {:reply, {:error, "Port communication error"}, port}
    end
  end

  def handle_info({_port, {:data, @term_call <> data}}, port) do
    msg = :erlang.binary_to_term(data)
    Logger.warn("Unexpected message from CUDA port: #{inspect msg}")
    {:noreply, port}
  end

  #def handle_info({:EXIT, from, :normal}, port) do
  #  IO.inspect("EXIT SIGNAL")
  #  if port != from do
  #    IO.inspect("EXIT NOT FROM PORT - TERMINATE PORT GRACEFULLY")
  #    unless is_nil(Port.info(port)), do: Port.close(port)
  #    # Port.close(port)
  #  end
  #  {:stop, :normal, port}
  #end
end
