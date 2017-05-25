defmodule Cuda.Worker do
  use GenServer
  alias Cuda.Shared
  alias Cuda.Graph
  alias Cuda.Graph.Node
  alias Cuda.Graph.GPUNode
  alias Cuda.Graph.Factory
  #alias Cuda.Graph.Processing
  alias Cuda.Compiler.Unit
  alias Cuda.Compiler.Context
  alias Cuda.Runner

  def start_link(opts \\ []) do
    {opts, args} = Keyword.split(opts, ~w(name)a)
    GenServer.start_link(__MODULE__, args, opts)
  end

  def init(opts) do
    with {:ok, module} <- Keyword.fetch(opts, :network),
         {:ok, cuda} <- Cuda.start_link(),
         {:ok, info} <- Cuda.device_info(cuda) do
      st = %{cuda: cuda, module: module, info: info,
             shared_pid: Keyword.get(opts, :shared_pid),
             graph: nil, shared: nil, shared_offsets: nil}
      load_network(st, opts)
    end
  end

  def run(pid, input) do
    GenServer.call(pid, {:run, input})
  end

  def gpu_info(pid) do
    GenServer.call(pid, :info)
  end

  def handle_call({:run, input}, _from, st) do
    opts = [cuda: st.cuda, args: %{shared: st.shared}]
    # [x] = st.graph.nodes
    # IO.inspect(x.assigns.batch)
    result = Runner.run(st.graph, input, opts)
    {:reply, result, st}
  end

  def handle_call(:info, _from, st) do
    {:reply, {:ok, st.info}, st}
  end

  defp load_network(st, opts) do
    env = %Cuda.Env{gpu_info: st.info}
    ctx = %Context{}
    nopts = Keyword.get(opts, :network_options, [])
    with {:module, _} <- Code.ensure_loaded(st.module),
         proto = struct(Node.proto(st.module)),
         %{} = graph <- Factory.new(proto, :network, st.module, nopts, env),
         {:ok, shared, offsets} <- load_shared(graph, st, opts),
         # compile sources into cubin
         {:ok, graph} <- Unit.compile(graph, %{ctx | assigns: %{shared_offsets: offsets}}),
         # load compiled cubins into GPU
         {:ok, graph} <- Runner.load(graph, shared: shared, cuda: st.cuda) do
      {:ok, %{st | graph: graph, shared: shared, shared_offsets: offsets}}
    end
  end

  defp load_shared(graph, st, opts) do
    with %{} = shared  <- collect_shared(graph),
         {:ok, shared} <- make_shared(shared, Keyword.get(opts, :shared, %{})),
         # load shared variables
         {:ok, _} <- Shared.load(st.shared_pid, shared),
         {:ok, shared} <- Shared.share(st.shared_pid),
         {:ok, shared} <- Cuda.memory_load(st.cuda, shared),
         # retrieve shared variables offset for compilation
         {:ok, offsets} <- Shared.offsets(st.shared_pid) do
      {:ok, shared, offsets}
    else
      _ -> {:ok, nil, %{}}
    end
  end

  defp make_shared(shared, values) do
    IO.inspect({shared, values})
    shared = shared |> Enum.reduce(%{}, fn {k, types}, m ->
      v = values
          #|> Map.get(k, %{})
          |> Enum.reduce(%{}, fn {name, values}, acc ->
            case Map.get(values, k) do
              nil -> acc
              values -> Map.put(acc, name, values)
            end
          end)
          #|> Enum.into(%{})
      Map.put(m, k, {types, v})
    end) |> IO.inspect
    case length(Map.keys(shared)) do
      0 -> :empty_shared
      _ -> {:ok, shared}
    end
  end

  defp merge_shared(shared, _, nil), do: shared
  defp merge_shared(shared, id, values) do
    id = Node.string_id(id)
    values |> Enum.reduce(shared, fn {key, value}, shared ->
      Map.put(shared, key, Map.put(Map.get(shared, key, %{}), id, value))
    end)
  end

  #defp nest_shared(shared, id, nested) do
  #  nested |> Enum.reduce(shared, fn {key, values}, shared ->
  #    values = values
  #             |> Enum.map(fn {nid, v} -> {Node.string_id({id, nid}), v} end)
  #             |> Enum.into(%{})
  #    Map.put(shared, key, Map.merge(Map.get(shared, key, %{}), values))
  #  end)
  #end

  defp collect_shared(graph, root \\ true)
  defp collect_shared(%GPUNode{id: id, assigns: %{shared: shared}}, _) do
    {:ok, merge_shared(%{}, id, shared)}
  end
  defp collect_shared(%{id: _gid} = graph, _root) do
    graph.nodes |> Enum.reduce(Map.get(graph.assigns, :shared, %{}), fn
      %Graph{} = g, shared -> Map.merge(shared, collect_shared(g))
      %{assigns: assigns}, shared -> Map.merge(shared, Map.get(assigns, :shared, %{}))
    end)
    #Processing.dfs(graph, fn
    #  :enter, {%{id: ^gid, assigns: assigns}, _}, st ->
    #    if root do
    #      {:ok, merge_shared(st, gid, Map.get(assigns, :shared))}
    #    else
    #      {:ok, st}
    #    end
    #  :enter, {%Graph{id: id, assigns: assigns} = g, _}, st ->
    #    #id = Node.string_id(id)
    #    #with shared <- merge_shared(st, id, Map.get(assigns, :shared)),
    #    #     {:ok, nested} <- collect_shared(g, false) do
    #    #  {:ok, nest_shared(shared, id, nested)}
    #    #end
    #    shared = Map.merge(st, Map.get(assigns, :shared, %{}))
    #    Enum.reduce(g.nodes, )
    #  :enter, {%GPUNode{id: id, assigns: %{shared: shared}}, _}, st ->
    #    {:ok, merge_shared(st, id, shared)}
    #  _, _, st ->
    #    {:ok, st}
    #end, %{})
  end
end
