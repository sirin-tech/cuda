defimpl Cuda.Compiler.GPUUnit, for: Cuda.Graph do
  alias Cuda.Graph.{Node, Pin, NodeProto, GraphProto, Processing}
  alias Cuda.Compiler.{Context, GPUUnit}
  alias Cuda.Template.PtxHelpers

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
            |> collect_batches(ctx, chains)
    #IO.inspect(graph.assigns)
    #Cuda.Graph.Visualize.Dot.render(graph, output: "/tmp/t.svg")
    {:ok, graph}
  end
  def sources(graph, _), do: {:ok, graph}

  defp collect_chains(graph) do
    longest_chain = graph |> Processing.longest_path()
    longest = longest_chain |> Enum.map(&Tuple.to_list/1) |> List.flatten
    inputs = graph |> NodeProto.pins(input_pin_types()) |> Enum.map(& &1.id)
    st = %{chains: [Enum.reverse(longest_chain)], visited: [], longest: longest}
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
          src in st.longest and dst in st.longest ->
            {:ok, %{st | visited: [src_node | st.visited]}}
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
      |> Enum.reject(& &1 == [])
      |> Enum.reverse
      |> Enum.map(&Enum.reverse/1)
      |> Enum.map(& %{links: &1, groups: [], size1: 0, size2: 0})
    end
  end

  defp collect_sizes(%{links: links} = chain, graph) do
    # get all pins in the chain and add pin layout to links
    links = links |> Enum.map(fn {src, dst} ->
      src_pin = GraphProto.link_spec_pin(graph, src)
      dst_pin = GraphProto.link_spec_pin(graph, dst)
      {src, dst, src_pin.group || dst_pin.group}
    end)

    # sum sizes of all grouped pins
    groups = links
             |> Enum.reject(& is_nil(elem(&1, 2)))
             |> Enum.group_by(& elem(&1, 2))
             |> Enum.map(fn {k, links} ->
               links = links
               |> Enum.group_by(& elem(&1, 0))
               |> Enum.map(fn {src, _} ->
                 size = GraphProto.link_spec_pin(graph, src) |> Pin.data_size()
                 {src, size}
               end)
               {k, links}
             end)
    chain = %{chain | links: links, groups: groups}# |> IO.inspect

    # calculate max sizes for floating pins
    links
    |> Enum.chunk_by(& not is_nil(elem(&1, 2)))
    # remove groups
    |> Enum.reject(& length(&1) > 0 and not is_nil(elem(List.first(&1), 2)))
    # found max sizes for each group of floating pins
    |> Enum.reduce(chain, fn floating, chain ->
      floating
      |> Enum.map(&elem(&1, 0))
      |> Enum.map(&GraphProto.link_spec_pin(graph, &1))
      |> Enum.map(&Pin.data_size/1)
      |> Enum.with_index
      |> Enum.reduce(chain, fn
        {n, i}, chain when is_odd(i) -> %{chain | size2: max(n, chain.size2)}
        {n, _}, chain                -> %{chain | size1: max(n, chain.size1)}
      end)
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
    # get total size of groups pins in all chains
    groups = chains |> Enum.flat_map(& &1.groups)
    fixed_size = groups
                 |> Keyword.values()
                 |> Enum.flat_map(&Keyword.values/1)
                 |> Enum.reduce(0, &+/2)
    {groups, _} = groups
                  |> Enum.reduce({[], 0}, fn {group, links}, {groups, offset} ->
                    {links, offset} = links |> Enum.map_reduce(offset, fn {link, size}, offset ->
                      {{link, {offset, size}}, offset + size}
                    end)
                    links = Keyword.get(groups, group, []) ++ links
                    {Keyword.put(groups, group, links), offset}
                  end)
    {graph, _} = chains |> Enum.reduce({graph, fixed_size}, fn chain, {graph, floating_offset} ->
      # collect fixed pins offsets
      state = {graph, groups}
      {graph, _} = chain.links |> Enum.reduce(state, &group_offsets_reducer/2)
      # collect floating pins offsets
      state = {graph, floating_offset, floating_offset + chain.size1}
      {graph, _, _} = chain.links |> Enum.reduce(state, &floating_offsets_reducer/2)

      pin_size = chain.size1 + chain.size2
      size = Map.get(graph.assigns, :pin_size, 0) + pin_size
      {graph |> NodeProto.assign(:pin_size, size), floating_offset + pin_size}
    end)
    graph
  end

  defp group_offsets_reducer({src, dst, group}, {graph, groups}) when not is_nil(group) do
    #IO.inspect(groups[group] |> Enum.find(fn {{^src, ^dst}, _} -> true; _ -> false end))
    case groups[group] |> Enum.find(fn {^src, _} -> true; _ -> false end) do
      {_, {offset, _}} ->
        graph = graph
                |> put_offset(src, offset)
                |> put_offset(dst, offset)
        {graph, groups}
      _ ->
        compile_error("Can't find offset for link `#{inspect {src, dst}}`")
    end
  end
  defp group_offsets_reducer(_, state) do
    state
  end

  defp floating_offsets_reducer({src, dst, nil}, {graph, o1, o2}) do
    graph = graph |> put_offset(src, o1) |> put_offset(dst, o1)
    {graph, o2, o1}
  end
  defp floating_offsets_reducer(_, state) do
    state
  end

  defp collect_batches(graph, ctx, chains) do
    batches = chains |> Enum.map(fn
      %{links: [{{:__self__, _}, {dst, _}, _} = link]} ->
        if length(chains) == 1 do
          node = GraphProto.node(graph, dst)
          batch = Map.get(node.assigns, :batch, [])
          add_final_event(batch, [link])
        else
          []
          #if is_nil(group), do: [{:event, Node.string_id(dst)}], else: []
        end
      %{links: [{{:__self__, _}, _, _} | links]} ->
        batch = reduce_batches([], links, graph, chains)
        add_final_event(batch, links)
      %{links: [{{node_id, _}, {dst_id, _}, group} = link | links]} ->
        wait = case dependency(chains, node_id) do
          nil    -> compile_error("Can't find dependency for node #{Node.string_id(node_id)}")
          dep_id -> [{:copy, link}, {:wait, dep_id}]
        end
        batch = reduce_batches(wait, links, graph, chains)
        if batch == wait do
          if dst_id == :__self__ or not is_nil(group) do
            # Don't generate event on outputs or fixed pins
            batch
          else
            batch ++ [{:event, Node.string_id(dst_id)}]
          end
        else
          add_final_event(batch, links)
        end
    end)
    {batches, sources} = batches |> Enum.map_reduce([], fn batch, sources ->
      batch
      |> Enum.filter(fn
        {:copy, {_, {:__self__, _}, _}} ->
          # don't copy outputs
          false
        {:copy, {_, dst, group}} ->
          # don't copy pins that are fixed to groups
          with true <- is_nil(group) do
            # don't copy terminator pins
            pin = GraphProto.link_spec_pin(graph, dst)
            pin.type != :terminator
          end
        _ ->
          true
      end)
      |> Enum.map_reduce(sources, fn
        {:copy, link}, sources ->
          {name, ptx, size} = copy_ptx(graph, ctx, link)
          {{:run, {name, {1, 1, 1}, {size, 1, 1}, []}}, [{:ptx, ptx} | sources]}
        batch, sources -> {batch, sources}
      end)
    end)
    batches = batches |> Enum.filter(fn
                        [{:wait, _}] -> false
                        _            -> true
                      end)
                      |> Enum.filter(fn
                        [] -> false
                        _  -> true
                      end)
    sources = Map.get(graph.assigns, :sources, []) ++ sources
    NodeProto.assign(graph, batches: batches, sources: sources)
  end

  defp copy_ptx(graph, ctx, {src, dst, _}) do
    {{src_offset, _}, pin} = case src do
      {:__self__, pin} ->
        {graph.assigns.pin_offsets[pin], NodeProto.pin(graph, pin)}
      {node, pin} ->
        node = GraphProto.node(graph, node)
        {node.assigns.pin_offsets[pin], NodeProto.pin(node, pin)}
    end
    {dst_offset, _} = case dst do
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
    ptx = PtxHelpers.header(ctx) <> """
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
    links |> Enum.reduce(batch, fn {{node_id, _}, _, _}, batch ->
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
    with {_, {:__self__, _}, _} <- links |> List.last do
      batch
    else
      {_, {node_id, _}, nil} = link ->
        copy = if length(links) > 1, do: [{:copy, link}], else: []
        batch ++ copy ++ [{:event, Node.string_id(node_id)}]
      _ ->
        batch
    end
  end

  # find all nodes that are produces result for specified node
  defp dependencies(chains, node_id) do
    chains |> Enum.reduce([], fn %{links: links}, chains ->
      case links do
        [{{:__self__, _}, _, _}] ->
          chains
        links ->
          with {_, {^node_id, _}, nil} <- List.last(links) do
            chains ++ [Node.string_id(node_id)]
          else
            _ -> chains
          end
      end
    end)
  end

  # Find an id of node that is depended from specified node_id
  # (that is placed not at the chain start)
  defp dependency(chains, node_id) do
    chains |> Enum.find_value(fn
      %{links: [{{^node_id, _}, _, _} | _]} -> false
      %{links: links} ->
        Enum.find_value(links, fn
          {{^node_id, _}, _, _} -> Node.string_id(node_id)
          _                     -> false
        end)
    end)
  end

  # Find all dependend chains (thats starts with specified node id)
  defp dependend(chains, node_id) do
    chains |> Enum.reduce([], fn
      {%{links: [{{^node_id, _} = src, dst, nil} | _]}, _}, list ->
        [{dst, src} | list]
      _, list ->
        list
    end)
  end

  defp put_offset(graph, {_node, pin} = link_spec, offset) do
    node = GraphProto.link_spec_node(graph, link_spec)
    size = graph |> GraphProto.link_spec_pin(link_spec) |> Pin.data_size()
    offsets = node.assigns
              |> Map.get(:pin_offsets, %{})
              |> Map.put(pin, {offset, size})
    node = NodeProto.assign(node, :pin_offsets, offsets)
    if node.id == graph.id do
      node
    else
      GraphProto.replace(graph, node)
    end
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
              |> Enum.uniq()
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
