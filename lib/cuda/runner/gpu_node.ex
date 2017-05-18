defimpl Cuda.Runner, for: Cuda.Graph.GPUNode do
  alias Cuda.Graph.NodeProto
  alias Cuda.Graph.Pin
  import Cuda.Graph.Node, only: [output_pin_types: 0]

  defp make_binary(v) when is_list(v) do
    v |> List.flatten |> Enum.reduce(<<>>, & &2 <> <<&1::float-little-32>>)
  end

  def load(%{assigns: assigns} = node, opts) do
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
      {:ok, NodeProto.assign(node, cuda_module: module, cuda_args: args)}
    end
  end

  def run(%{assigns: assigns} = node, inputs, opts) do
    with cuda when is_pid(cuda) <- Keyword.get(opts, :cuda) do
      {pins, extracts} = Enum.reduce(assigns.offsets, {<<>>, %{}}, fn {k, offset}, {pins, extracts} ->
        pin = NodeProto.pin(node, k)
        size = Pin.data_size(pin)
        pins = case Map.get(inputs, k) do
          nil -> pins <> <<0::unit(8)-size(size)>>
          v   -> pins <> make_binary(v)
        end
        extracts = if pin.type in output_pin_types() do
          Map.put(extracts, k, {offset, size})
        else
          extracts
        end
        {pins, extracts}
      end)

      {:ok, mpins}  = Cuda.memory_load(cuda, pins)

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

      :ok = Cuda.stream(cuda, assigns.cuda_module, batch)
      {:ok, o} = Cuda.memory_read(cuda, mpins)

      outputs = Enum.reduce(extracts, %{}, fn {k, extract}, acc ->
        v = :binary.part(o, extract)
        v = (for <<x::float-little-32 <- v>>, do: x)
            |> Enum.map(& Float.round(&1, 1))
        Map.put(acc, k, v)
      end)
      {:ok, outputs}
    else
      _ -> {:error, :no_cuda_specified}
    end
  end
end
