defimpl Cuda.Compiler.GPUUnit, for: Cuda.Graph do
  alias Cuda.Graph.{Node, Pin, NodeProto, GraphProto, Processing}
  alias Cuda.Compiler.{Context, GPUUnit}

  require Integer
  require Cuda
  import Integer, only: [is_odd: 1]
  import Cuda, only: [compile_error: 1]
  import Node, only: [input_pin_types: 0]
  import Cuda.Compiler.Utils

  def sources(%{type: :computation_graph} = graph, ctx) do
    chains = graph
             |> collect_chains()
             |> Enum.map(& collect_sizes(&1, graph))
    graph = graph
            |> collect_offsets(chains)
            |> put_pins_shapes()
            |> collect_sources(ctx)
            |> collect_batches(chains)
    #IO.inspect(graph.assigns)
    #Cuda.Graph.Visualize.Dot.render(graph, output: "/tmp/t.svg")
    {:ok, graph}
  end
  def sources(graph, _), do: {:ok, graph}

  defp collect_chains(graph) do
    longest_chain = graph |> Processing.longest_path()
    longest = longest_chain |> Enum.map(&Tuple.to_list/1) |> List.flatten
    inputs = graph |> NodeProto.pins(input_pin_types()) |> Enum.map(& &1.id)
    st = %{chains: [longest_chain], visited: [], longest: longest}
    result = Processing.dfs(graph, fn
      # graph input
      :enter, {:__self__, pin} = src, st ->
        cond do
          # if not an input - skip
          not pin in inputs -> {:ok, st}
          # if already in longest chain - skip it
          src in st.longest -> {:ok, st}
          # if node not in longest chain - start new chain
          true -> {:ok, %{st | chains: [[] | st.chains]}}
        end
      # move from one node to another
      :move, {{src_node, _} = src, dst} = link, %{chains: [chain | chains]} = st ->
        cond do
          # if link already in longest - skip it
          src in st.longest or dst in st.longest ->
            {:ok, st}
          # if we move from already visited node then this node have more that
          # one output and we should start new chain
          src_node in st.visited ->
            {:ok, %{st | chains: [[link] | st.chains]}}
          # add link to current chain
          true ->
            {:ok, %{st | chains: [[link | chain] | chains],
                         visited: [src_node | st.visited]}}
        end
      _, _, st ->
        {:ok, st}
    end, st, ids: true)
    with {:ok, st} <- result do
      st.chains
      |> Enum.reverse
      |> Enum.map(& %{links: &1, size1: 0, size2: 0})
    end
  end

  defp collect_sizes(%{links: links} = chain, graph) do
    links
    |> Enum.map(fn
      {{:__self__, pin}, _} ->
        graph |> NodeProto.pin(pin) |> Pin.data_size()
      {{node, pin}, _} ->
        graph |> GraphProto.node(node) |> NodeProto.pin(pin) |> Pin.data_size()
    end)
    |> Enum.with_index
    |> Enum.reduce(chain, fn
      {n, i}, chain when is_odd(i) -> %{chain | size2: max(n, chain.size2)}
      {n, _}, chain                -> %{chain | size1: max(n, chain.size1)}
    end)
  end

  defp collect_sources(graph, ctx) do
    ctx = Context.replace_current(ctx, graph)
    graph.nodes |> Enum.reduce(graph, fn node, graph ->
      with {:ok, node} <- node.module.__compile__(node),
           {:ok, node} <- GPUUnit.sources(node, Context.for_node(ctx, node)) do
        id = Node.string_id(node.id)
        batch = node.module.__batch__(node) |> Enum.map(fn
          {:run, {name, g, b, args}} -> {:run, {"#{id}__#{name}", g, b, args}}
          {:run, {name, g, b}}       -> {:run, {"#{id}__#{name}", g, b, []}}
        end)
        sources = Map.get(graph.assigns, :sources, []) ++ node.assigns.sources
        node = NodeProto.assign(node, :batch, batch)
        graph
        |> NodeProto.assign(:sources, sources)
        |> GraphProto.replace(node)
      else
        _ -> graph
      end
    end)
  end

  defp collect_offsets(%{} = graph, chains) do
    {graph, _} = chains |> Enum.reduce({graph, 0}, fn chain, {graph, chain_offset} ->
      state = {graph, chain_offset, chain_offset + chain.size1}
      {graph, _, _} = chain.links |> Enum.reduce(state, fn
        {{:__self__, src_pin}, {dst, dst_pin}}, {graph, o1, o2} ->
          node = GraphProto.node(graph, dst) |> put_offset(dst_pin, o1)
          graph = graph |> put_offset(src_pin, o1) |> GraphProto.replace(node)
          {graph, o2, o1}
        {{src, src_pin}, {:__self__, dst_pin}}, {graph, o1, o2} ->
          node = GraphProto.node(graph, src) |> put_offset(src_pin, o1)
          graph = graph |> put_offset(dst_pin, o1) |> GraphProto.replace(node)
          {graph, o2, o1}
        {{:__self__, src_pin}, {:__self__, dst_pin}}, {graph, o1, o2} ->
          graph = graph |> put_offset(src_pin, o1) |> put_offset(dst_pin, o1)
          {graph, o2, o1}
        {{src, src_pin}, {dst, dst_pin}}, {graph, o1, o2} ->
          src = GraphProto.node(graph, src) |> put_offset(src_pin, o1)
          dst = GraphProto.node(graph, dst) |> put_offset(dst_pin, o1)
          graph = graph |> GraphProto.replace(src) |> GraphProto.replace(dst)
          {graph, o2, o1}
      end)
      pin_size = chain.size1 + chain.size2
      size = Map.get(graph.assigns, :pin_size, 0) + pin_size
      {graph |> NodeProto.assign(:pin_size, size), chain_offset + pin_size}
    end)
    graph
  end

  defp collect_batches(graph, chains) do
    batches = chains |> Enum.map(fn
      %{links: [{{:__self__, _}, {dst, _}} = link]} ->
        [{:copy, link}, {:event, Node.string_id(dst)}]
      %{links: [{{:__self__, _}, _} | links]} ->
        batch = reduce_batches([], links, graph, chains)
        add_final_event(batch, links)
      %{links: [{{node_id, _}, _} | links]} ->
        batch = case dependency(chains, node_id) do
          nil    -> compile_error("Can't find dependency for node #{node_id}")
          dep_id -> [{:wait, dep_id}]
        end
        reduce_batches(batch, links, graph, chains)
        add_final_event(batch, links)
    end)
    {batches, sources} = batches |> Enum.map_reduce([], fn batch, sources ->
      batch |> Enum.map_reduce(sources, fn
        {:copy, link}, sources ->
          {name, ptx, size} = copy_ptx(graph, link)
          {{:run, {name, {1, 1, 1}, {size, 1, 1}, []}}, [{:ptx, ptx} | sources]}
        batch, sources -> {batch, sources}
      end)
    end)
    sources = Map.get(graph.assigns, :sources, []) ++ sources
    NodeProto.assign(graph, batches: batches, sources: sources)
  end

  defp copy_ptx(graph, {src, dst}) do
    {src_offset, pin} = case src do
      {:__self__, pin} ->
        {graph.assigns.pin_offsets[pin], NodeProto.pin(graph, pin)}
      {node, pin} ->
        node = GraphProto.node(graph, node)
        {node.assigns.pin_offsets[pin], NodeProto.pin(node, pin)}
    end
    dst_offset = case dst do
      {:__self__, pin} ->
        graph.assigns.pin_offsets[pin]
      {node, pin} ->
        node = GraphProto.node(graph, node)
        node.assigns.pin_offsets[pin]
    end
    type = Pin.data_type(pin)
    size = Pin.data_size(pin)
    type_size = div(size, Pin.data_arity(pin))
    name = "copy_#{UUID.uuid1()}" |> String.replace("-", "")
    ptx = """
    .version 5.0
    .target sm_30
    .address_size 64
    .visible .entry #{name} (.param .u64 .ptr pins) {
      .reg .u64 %pins, %out;
      .reg .u32 %x;
      .reg .#{type} %f;
      ld.param.u64 %pins, [pins];
      mov.u32 %x, %ctaid.x;
      mad.wide.u32 %pins, %x, #{type_size}, %pins;
      add.u64 %out, %pins, #{type_size * dst_offset};
      add.u64 %pins, %pins, #{type_size * src_offset};
      ld.global.#{type} %f, [%pins];
      st.global.#{type} [%out], %f;
      ret;
    }
    """# |> IO.puts
    {name, ptx, Pin.data_arity(pin)}
  end

  defp reduce_batches(batch, links, graph, chains) do
    links |> Enum.reduce(batch, fn {{node_id, _}, _}, batch ->
      deps = chains |> dependencies(node_id) |> Enum.map(& {:wait, &1})
      node = GraphProto.node(graph, node_id)
      batch = batch ++ deps ++ Map.get(node.assigns, :batch, [])
      case chains |> dependend(node_id) do
        []   -> batch
        list -> batch ++ Enum.map(list, & {:copy, &1}) ++ [{:event, Node.string_id(node_id)}]
      end
    end)
  end

  defp add_final_event(batch, links) do
    with {_, {:__self__, _}} <- links |> List.last do
      batch
    else
      {_, {node_id, _}} = link ->
        batch ++ [{:copy, link}, {:event, Node.string_id(node_id)}]
    end
  end

  defp dependencies(chains, node_id) do
    chains |> Enum.reduce([], fn %{links: links}, chains ->
      with {_, {^node_id, _}} <- List.last(links) do
        chains ++ [Node.string_id(node_id)]
      else
        _ -> chains
      end
    end)
  end

  defp dependency(chains, node_id) do
    chains |> Enum.find_value(fn
      %{links: [{{^node_id, _}, _} | _]} -> false
      %{links: links} ->
        Enum.find_value(links, fn
          {{^node_id, _}, _} -> Node.string_id(node_id)
          _                  -> false
        end)
    end)
  end

  defp dependend(chains, node_id) do
    chains |> Enum.reduce([], fn
      {%{links: [{{^node_id, _} = src, dst} | _]}, _}, list -> [{dst, src} | list]
      _, list -> list
    end)
  end

  defp put_offset(%{assigns: assigns} = node, pin, offset) do
    offsets = assigns |> Map.get(:pin_offsets, %{}) |> Map.put(pin, offset)
    NodeProto.assign(node, :pin_offsets, offsets)
  end
end

defimpl Cuda.Compiler.Unit, for: Cuda.Graph do
  alias Cuda.Compiler
  alias Cuda.Compiler.{Context, GPUUnit}
  alias Cuda.Graph.{NodeProto, GraphProto, Processing}
  require Logger

  def compile(%{type: :computation_graph} = graph, ctx) do
    Logger.info("CUDA: Compiling GPU code for graph #{graph.module} (#{graph.id})")
    with {:ok, graph} <- graph.module.__compile__(graph),
         {:ok, graph} <- GPUUnit.sources(graph, ctx) do
      if Map.get(ctx.assigns, :compile_sources) != false do
        with {:ok, cubin} <- Compiler.compile(graph.assigns.sources) do
          {:ok, NodeProto.assign(graph, :cubin, cubin)}
        end
      else
        {:ok, graph}
      end
    else
      _ ->
        Logger.warn("CUDA: Error while compiling GPU code for graph " <>
                    "#{graph.module} (#{graph.id})")
        {:error, :compile_error}
    end
  end
  def compile(graph, ctx) do
    Logger.info("CUDA: Compiling graph #{graph.module} (#{graph.id})")
    with {:ok, graph}    <- graph.module.__compile__(graph),
         Cuda.Graph.Visualize.Dot.render(graph, output: "/tmp/source.svg"),
         %{} = graph     <- Processing.expand(graph),
         Cuda.Graph.Visualize.Dot.render(graph, output: "/tmp/expanded.svg"),
         %{} = graph     <- Processing.precompile_wrap(graph),
         Cuda.Graph.Visualize.Dot.render(graph, output: "/tmp/wrapped.svg"),
         %{} = graph     <- topology_sort(graph),
         ctx = Context.for_node(ctx, graph),
         {:ok, _, nodes} <- Enum.reduce(graph.nodes, {:ok, ctx, []}, &compile_reducer/2) do
      assigns = Enum.reduce(nodes, graph.assigns, fn
        %{type: :computation_graph, assigns: %{pin_offsets: offsets}}, acc ->
          offsets = offsets
                    |> Enum.map(fn {pin, value} ->
                      {convert_pin_name(pin, graph.links), value}
                    end)
                    |> Enum.into(%{})
          Map.put(acc, :pin_offsets, offsets)
        _, acc ->
          acc
      end)
      {:ok, %{graph | nodes: nodes, assigns: assigns}}
    else
      _ ->
        Logger.warn("CUDA: Error while compiling graph " <>
                    "#{graph.module} (#{graph.id})")
        {:error, :compile_error}
    end
  end

  defp topology_sort(%{id: gid} = graph) do
    with {:ok, nodes} <- Processing.topology_sort(graph) do
      nodes = nodes
              |> Enum.map(fn {node, _pin} -> node end)
              |> Enum.reject(& &1 == gid)
              |> Enum.map(& GraphProto.node(graph, &1))
      %{graph | nodes: nodes}
    else
      _ -> graph
    end
  end

  defp compile_reducer(node, {:ok, ctx, nodes}) do
    ctx = Context.for_node(ctx, node)
    with {:ok, node} <- Cuda.Compiler.Unit.compile(node, ctx) do
      {:ok, ctx, [node | nodes]}
    end
  end
  defp compile_reducer(_, error) do
    error
  end

  defp convert_pin_name(id, links) do
    Enum.find_value(links, fn
      {{:__self__, pin}, {_, ^id}} -> pin
      {{_, ^id}, {:__self__, pin}} -> pin
      _ -> nil
    end) || id
  end
end

defimpl Cuda.Compiler.Unit, for: Cuda.Graph.Node do
  require Logger

  def compile(node, ctx) do
    Logger.info("CUDA: Compiling node #{node.module} (#{node.id})")
    node.module.__compile__(node, ctx)
  end
end
