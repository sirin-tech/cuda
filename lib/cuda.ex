defmodule Cuda do
  @moduledoc """
  NVIDIA GPU CUDA library bindings for Erlang and Elixir.
  """
  use GenServer
  require Logger

  @term_call <<1>>
  @raw_call  <<2>>
  @proxied_calls ~w(call call_raw)a

  @type error_tuple :: {:error, String.t}

  defmacro __using__(_opts) do
    quote do
      use GenServer
      require Logger

      defdelegate info(pid), to: unquote(__MODULE__)
      defdelegate info(pid, info), to: unquote(__MODULE__)
      defdelegate compile(pid, src), to: unquote(__MODULE__)
      defdelegate compile(pid, src, opts), to: unquote(__MODULE__)
      defdelegate module_load(pid, src), to: unquote(__MODULE__)
      defdelegate module_load(pid, src, opts), to: unquote(__MODULE__)
      defdelegate memory_load(pid, data), to: unquote(__MODULE__)
      defdelegate module_read(pid, handle), to: unquote(__MODULE__)
      defdelegate module_unload(pid, handle), to: unquote(__MODULE__)
      defdelegate module_share(pid, handle), to: unquote(__MODULE__)
      defdelegate run(pid, module, func, params), to: unquote(__MODULE__)
      defdelegate run(pid, module, func, block, params), to: unquote(__MODULE__)
      defdelegate run(pid, module, func, block, grid, params), to: unquote(__MODULE__)
      defdelegate stream(pid, module, batch), to: unquote(__MODULE__)
      defdelegate start_link(), to: unquote(__MODULE__)
      defdelegate start_link(opts), to: unquote(__MODULE__)

      def __init__(opts), do: {:ok, opts}
      def __handle_call__(_msg, _from, st), do: {:noreply, st}
      def __handle_cast__(_msg, st), do: {:noreply, st}
      def __handle_into__(_msg, st), do: {:noreply, st}

      def init(opts) do
        with {:ok, proxy_st} <- unquote(__MODULE__).init(opts),
             {:ok, st} <- __init__(opts) do
          {:ok, {st, proxy_st}}
        end
      end

      def handle_call({x, _, _} = msg, from, {st, proxy_st}) when x in unquote(@proxied_calls) do
        case unquote(__MODULE__).handle_call(msg, from, proxy_st) do
          {:reply, reply, proxy_st} -> {:reply, reply, {st, proxy_st}}
          {:reply, reply, proxy_st, timeout} -> {:reply, reply, {st, proxy_st}, timeout}
          {:noreply, proxy_st} -> {:noreply, {st, proxy_st}}
          {:noreply, proxy_st, timeout} -> {:noreply, {st, proxy_st}, timeout}
          {:stop, reason, reply, proxy_st} -> {:stop, reason, reply, {st, proxy_st}}
          {:stop, reason, proxy_st} -> {:stop, reason, {st, proxy_st}}
        end
      end
      def handle_call(msg, from, {st, proxy_st}) do
        case __handle_call__(msg, from, st) do
          {:reply, reply, st} -> {:reply, reply, {st, proxy_st}}
          {:reply, reply, st, timeout} -> {:reply, reply, {st, proxy_st}, timeout}
          {:noreply, st} -> {:noreply, {st, proxy_st}}
          {:noreply, st, timeout} -> {:noreply, {st, proxy_st}, timeout}
          {:stop, reason, reply, st} -> {:stop, reason, reply, {st, proxy_st}}
          {:stop, reason, st} -> {:stop, reason, {st, proxy_st}}
        end
      end

      def handle_cast(msg, {st, proxy_st}) do
        case __handle_cast__(msg, st) do
          {:noreply, st} -> {:noreply, {st, proxy_st}}
          {:noreply, st, timeout} -> {:noreply, {st, proxy_st}, timeout}
          {:stop, reason, st} -> {:stop, reason, {st, proxy_st}}
        end
      end

      def handle_info({_port, {:data, unquote(@term_call) <> _}} = msg, {st, proxy_st}) do
        case unquote(__MODULE__).handle_info(msg, proxy_st) do
          {:noreply, proxy_st} -> {:noreply, {st, proxy_st}}
          {:noreply, proxy_st, timeout} -> {:noreply, {st, proxy_st}, timeout}
          {:stop, reason, proxy_st} -> {:stop, reason, {st, proxy_st}}
        end
      end
      def handle_info(msg, {st, proxy_st}) do
        case __handle_info__(msg, st) do
          {:noreply, st} -> {:noreply, {st, proxy_st}}
          {:noreply, st, timeout} -> {:noreply, {st, proxy_st}, timeout}
          {:stop, reason, st} -> {:stop, reason, {st, proxy_st}}
        end
      end

      defoverridable __init__: 1, __handle_call__: 3, __handle_cast__: 2,
                     __handle_info__: 2
    end
  end

  def start_link(opts \\ []) do
    {name, opts} = Keyword.split(opts, ~w(name)a)
    GenServer.start_link(__MODULE__, opts, name)
  end

  defmacro compile_error(msg) do
    quote do
      raise CompileError, description: unquote(msg)
    end
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

  def device_info(pid) do
    GenServer.call(pid, {:call, :device_info, nil})
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

  def memory_load(pid, data) when is_binary(data) do
    GenServer.call(pid, {:call_raw, :memory_load, data})
  end
  def memory_load(pid, data) do
    GenServer.call(pid, {:call, :memory_load, data})
  end

  def memory_read(pid, handle) do
    GenServer.call(pid, {:call, :memory_read, handle})
  end

  def memory_unload(pid, handle) do
    GenServer.call(pid, {:call, :memory_unload, handle})
  end

  def memory_share(pid, handle) do
    GenServer.call(pid, {:call, :memory_share, handle})
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
  # NOTE: change @lint attribute to something that will be proposed to
  #       replace deprecation and activate Credo.Check.Refactor.FunctionArity
  #       in .credo.exs with default max_arity (5)
  # @lint {Credo.Check.Refactor.FunctionArity, false}
  def run(pid, module, func, block, grid, params) do
    GenServer.call(pid, {:call, :run, {module, func, block, grid, params}})
  end

  def stream(pid, module, batch) do
    GenServer.call(pid, {:call, :stream, {module, batch}})
  end

  def init(opts) do
    cmd = case Keyword.get(opts, :port_bin) do
      nil  -> Application.app_dir(:cuda, Path.join(~w(priv cuda_driver_port)))
      port -> port
    end
    cmd = case Keyword.get(opts, :device) do
      nil    -> cmd
      device -> "#{cmd} #{device}"
    end
    port = Port.open({:spawn, cmd}, [:binary, :nouse_stdio, packet: 4])
    {:ok, port}
  end

  def handle_call({:call, func, arg}, _from, port) do
    Port.command(port, @term_call <> :erlang.term_to_binary({func, arg}))
    wait_reply(port)
  end

  def handle_call({:call_raw, func, arg}, _from, port) do
    func = "#{func}"
    size = byte_size(func)
    Port.command(port, @raw_call <> <<size>> <> func <> arg)
    wait_reply(port)
  end

  defp wait_reply(port) do
    receive do
      {^port, {:data, @term_call <> data}} ->
        {:reply, :erlang.binary_to_term(data), port}
      {^port, {:data, @raw_call <> data}} ->
        {:reply, {:ok, data}, port}
      _ ->
        {:reply, {:error, "Port communication error"}, port}
    end
  end

  def handle_info({_port, {:data, @term_call <> data}}, port) do
    msg = :erlang.binary_to_term(data)
    Logger.warn("Unexpected message from CUDA port: #{inspect msg}")
    {:noreply, port}
  end
end
