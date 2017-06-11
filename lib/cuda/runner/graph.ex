defimpl Cuda.Runner, for: Cuda.Graph do
  alias Cuda.Graph.NodeProto
  alias Cuda.Memory

  import Cuda.Graph.Node, only: [input_pin_types: 0]

  def load(%{type: :computation_graph, assigns: assigns} = graph, opts) do
    with cuda when not is_nil(cuda) <- Keyword.get(opts, :cuda) do
      # load cubin into GPU
      {:ok, module} = Cuda.module_load(cuda, assigns.cubin)
      # load args into GPU
      args = opts
             |> Keyword.get(:args, %{})
             |> Enum.reduce(%{}, fn
               {k, {m, _} = loaded}, args when m in ~w(memory shared_memory)a ->
                 Map.put(args, k, loaded)
               {k, {type, value}}, args ->
                 bin = Memory.pack(type, value)
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

  def run(%{type: :computation_graph, assigns: assigns}, inputs, opts) do
    with cuda when not is_nil(cuda) <- Keyword.get(opts, :cuda) do
      # get input and convert it to binary
      pins = Memory.pack(inputs, assigns.memory.pins)# |> IO.inspect)
      # load pins into GPU
      {:ok, mpins} = Cuda.memory_load(cuda, pins)
      # prepare arguments and batch list
      args = Map.merge(Map.get(assigns, :cuda_args, %{}),
                       Keyword.get(opts, :args, %{}))
      batches = assigns.batches |> Enum.map(fn batch ->
        Enum.map(batch, fn
          {:run, {name, k, b, params}} ->
            params = Enum.map(params, & Map.get(args, &1))
            {:run, {name, k, b, [mpins | params]}}
          {:run, {name, k, b}} ->
            {:run, {name, k, b, [mpins]}}
          x ->
            x
        end)
      end)
      # run computation on GPU
      #IO.inspect({assigns.cuda_module, batches})
      :ok = Cuda.stream(cuda, assigns.cuda_module, batches)
      {:ok, pins} = Cuda.memory_read(cuda, mpins)
      #IO.inspect(for <<x::float-little-32 <- pins>>, do: x)
      #IO.inspect(byte_size(pins))
      output = pins |> Memory.unpack(assigns.memory.pins)
      {:ok, output}
    else
      _ -> {:error, :no_cuda_specified}
    end
  end
  def run(graph, inputs, opts) do
    pins = graph.links |> Enum.reduce(%{}, fn
      {{:__self__, input}, {dst, pin}}, pins ->
        node_pins = Map.get(pins, dst, %{})
        node_pins = Map.put(node_pins, pin, Map.get(inputs, input))
        Map.put(pins, dst, node_pins)
      _, pins ->
        pins
    end)
    pins = graph.nodes |> Enum.reduce({:ok, pins}, fn
      %{id: id} = node, {:ok, pins} ->
        inputs = node
                 |> NodeProto.pins(input_pin_types())
                 |> Enum.map(& &1.id)
                 |> Enum.into(MapSet.new)
        data = Map.get(pins, node.id, %{})
        available = data |> Map.keys() |> Enum.into(MapSet.new)
        #IO.inspect(MapSet.difference(inputs, available) |> MapSet.to_list(), label: CHECK)
        with [] <- MapSet.difference(inputs, available) |> MapSet.to_list() do
          inputs = data |> Map.take(MapSet.to_list(inputs))
          with {:ok, outputs} <- Cuda.Runner.run(node, inputs, opts) do
            #data = Map.merge(data, outputs)
            pins = graph.links |> Enum.reduce(pins, fn
              {{^id, output}, {dst, pin}}, pins ->
                node_pins = Map.get(pins, dst, %{})
                node_pins = Map.put(node_pins, pin, Map.get(outputs, output))
                Map.put(pins, dst, node_pins)
              _, pins ->
                pins
            end)
            {:ok, pins}
          end
        else
          # if not all inputs are ready - skip node
          _ -> {:error, "Not all inputs available. Possible graph loop"}
        end
      _, error ->
        error
    end)
    with {:ok, pins} <- pins do
      {:ok, Map.get(pins, :__self__)}
    end
  end
end
