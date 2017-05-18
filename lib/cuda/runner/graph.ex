defimpl Cuda.Runner, for: Cuda.Graph do
  alias Cuda.Graph.Processing
  alias Cuda.Graph.NodeProto
  alias Cuda.Graph.Pin
  import Cuda.Graph.Node, only: [input_pin_types: 0, output_pin_types: 0]

  defp make_binary(v) when is_list(v) do
    v |> List.flatten |> Enum.reduce(<<>>, & &2 <> <<&1::float-little-32>>)
  end

  def load(%{type: :computation_graph, assigns: assigns} = graph, opts) do
    with cuda when is_pid(cuda) <- Keyword.get(opts, :cuda) do
      # load cubin into GPU
      {:ok, module} = Cuda.module_load(cuda, assigns.cubin)
      # load args into GPU
      args = opts
             |> Keyword.get(:args, %{})
             |> Enum.reduce(%{}, fn
               {k, {m, _} = loaded}, args when m in ~w(memory shared_memory)a ->
                 Map.put(args, k, loaded)
               {k, {type, value}}, args ->
                 bin = Pin.pack(type, value)
                 with {:ok, marg} <- Cuda.memory_load(cuda, bin) do
                   Map.put(args, k, marg)
                 else
                   _ ->
                     # TODO: warning here
                     args
                 end
                _, args ->
                  args
             end)
      {:ok, NodeProto.assign(graph, cuda_module: module, cuda_args: args)}
    end
  end
  def load(%{nodes: nodes} = graph, opts) do
    nodes = nodes |> Enum.reduce([], fn node, nodes ->
      with {:ok, loaded} <- Cuda.Runner.load(node, opts) do
        [loaded] ++ nodes
      else
        _ -> [node] ++ nodes
      end
    end)
    {:ok, %{graph | nodes: nodes}}
  end

  def run(%{type: :computation_graph, assigns: assigns} = graph, inputs, opts) do
    with cuda when is_pid(cuda) <- Keyword.get(opts, :cuda) do
      # get input and convert it to binary
      input = NodeProto.pins(graph, input_pin_types()) |> List.first
      i = Map.get(inputs, input.id) |> make_binary
      # form pins layout
      size = assigns.pin_size - byte_size(i)
      pins = i <> <<0::unit(8)-size(size)>>
      # load pins into GPU
      {:ok, mpins} = Cuda.memory_load(cuda, pins)
      # prepare arguments and batch list
      args = Map.merge(Map.get(assigns, :cuda_args, %{}),
                       Keyword.get(opts, :args, %{}))
      batch = assigns.batch |> Enum.map(fn
        {name, k, b, params} ->
          params = Enum.map(params, & Map.get(args, &1))
          {name, k, b, [mpins | params]}
        {name, k, b} ->
          {name, k, b, [mpins]}
        x ->
          x
      end)
      # run computation on GPU
      #IO.inspect({assigns.cuda_module, batch})
      :ok = Cuda.stream(cuda, assigns.cuda_module, batch)
      {:ok, o} = Cuda.memory_read(cuda, mpins)
      # extract output
      output = NodeProto.pins(graph, output_pin_types()) |> List.first
      offset = Map.get(assigns.pin_offsets, output.id)
      size = Pin.data_size(output)
      <<_::unit(8)-size(offset), o::binary-size(size), _::binary>> = o
      # convert output from binary to float
      o = for <<x::float-little-32 <- o>>, do: x
      # return output
      {:ok, Map.put(%{}, output.id, o)}
    else
      _ -> {:error, :no_cuda_specified}
    end
  end
  def run(%{id: gid} = graph, inputs, opts) do
    st = %{pins: %{}, output: %{}}
    result = Processing.dfs(graph, fn
      # move from input pin to node - copy graph input to node input
      :move, {{%{id: ^gid}, src_pin}, {dst_node, dst_pin}}, st ->
        data = st.pins
               |> Map.get(dst_node.id, %{})
               |> Map.put(dst_pin.id, Map.get(inputs, src_pin.id))
        {:ok, %{st | pins: Map.put(st.pins, dst_node.id, data)}}
      # move from node to output pin - copy node output to resulting output
      :move, {{src_node, src_pin}, {%{id: ^gid}, dst_pin}}, st ->
        data = st.pins |> Map.get(src_node.id, %{}) |> Map.get(src_pin.id)
        {:ok, %{st | output: Map.put(st.output, dst_pin.id, data)}}
      # move from one node to another - copy src output to dst input
      :move, {{src_node, src_pin}, {dst_node, dst_pin}}, st ->
        src = Map.get(st.pins, src_node.id, %{})
        data = st.pins
               |> Map.get(dst_node.id, %{})
               |> Map.put(dst_pin.id, Map.get(src, src_pin.id))
        {:ok, %{st | pins: Map.put(st.pins, dst_node.id, data)}}
      # enter self pin - skip
      :enter, {%{id: ^gid}, _}, st ->
        {:ok, st}
      # enter node - run it and save results
      :enter, {node, _}, st ->
        # check if all inputs are available
        inputs = node
                 |> NodeProto.pins(input_pin_types())
                 |> Enum.map(& &1.id)
                 |> Enum.into(MapSet.new)
        data = Map.get(st.pins, node.id, %{})
        available = data |> Map.keys() |> Enum.into(MapSet.new)
        with [] <- MapSet.difference(inputs, available) |> MapSet.to_list() do
          inputs = data |> Map.take(MapSet.to_list(inputs))
          with {:ok, outputs} <- Cuda.Runner.run(node, inputs, opts) do
            data = Map.merge(data, outputs)
            {:ok, %{st | pins: Map.put(st.pins, node.id, data)}}
          end
        else
          # if not all inputs are ready - skip node
          _ -> {:ok, st}
        end
      # any other actions - just skip
      _, _, st ->
        {:ok, st}
    end, st)
    with {:ok, %{output: output}} <- result do
      {:ok, output}
    else
      _ -> {:error, :run_error}
    end
  end
end
