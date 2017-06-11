defmodule Cuda.Worker do
  use GenServer
  alias Cuda.{Runner, Shared}
  alias Cuda.Compiler.{Context, Unit}

  def start_link(opts \\ []) do
    {opts, args} = Keyword.split(opts, ~w(name)a)
    GenServer.start_link(__MODULE__, args, opts)
  end

  def init(opts) do
    cuda = case Keyword.get(opts, :cuda) do
      nil  -> with {:ok, cuda} <- Cuda.start_link(), do: cuda
      cuda -> cuda
    end
    with {:ok, graph} <- Keyword.fetch(opts, :graph) do
         #{:ok, info} <- Cuda.device_info(cuda) do
      st = %{cuda: cuda, graph: graph}
      load_graph(st, opts)
    end
  end

  def run(pid, input, args \\ %{}) do
    GenServer.call(pid, {:run, input, args})
  end

  #def gpu_info(pid) do
  #  GenServer.call(pid, :info)
  #end

  def handle_call({:run, input, args}, _from, st) do
    opts = [cuda: st.cuda, args: args]
    result = Runner.run(st.graph, input, opts)
    {:reply, result, st}
  end

  #def handle_call(:info, _from, st) do
  #  {:reply, {:ok, st.info}, st}
  #end

  defp load_graph(st, opts) do
    env = Keyword.get(opts, :env, %Cuda.Env{})
    env = with {:ok, info} <- Cuda.device_info(st.cuda) do
      %{env | gpu_info: info}
    else
      _ -> env
    end
    vars = %{float_size: env.float_size, f: Cuda.Env.f(env)}
    ctx = %Context{env: env, assigns: %{vars: vars}}
    args = case Keyword.get(opts, :shared) do
      nil ->
        %{}
      shared ->
        shared
        |> Enum.map(fn {k, pid} ->
          with {:ok, ref} <- Shared.share(pid),
               {:ok, mem} <- Cuda.memory_load(st.cuda, ref) do
            {k, mem}
          end
        end)
        |> Enum.into(%{})
    end
    with {:ok, graph} <- Unit.compile(st.graph, ctx),
         # load compiled cubins into GPU
         {:ok, graph} <- Runner.load(graph, args: args, cuda: st.cuda) do
      {:ok, %{st | graph: graph}}
    end
  end
end
