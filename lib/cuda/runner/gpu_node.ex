defimpl Cuda.Runner, for: Cuda.Graph.GPUNode do
  alias Cuda.Graph.NodeProto
  alias Cuda.Graph.Pin

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

  def run(%{assigns: assigns}, inputs, opts) do
    with cuda when is_pid(cuda) <- Keyword.get(opts, :cuda) do
      pins = inputs
             |> Cuda.Compiler.Utils.wrap_pins
             |> Pin.pack(assigns.inputs_shape)

      {:ok, mpins} = Cuda.memory_load(cuda, pins)

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

      :ok = Cuda.stream(cuda, assigns.cuda_module, batches)
      {:ok, pins} = Cuda.memory_read(cuda, mpins)

      output = pins
               |> Pin.unpack(assigns.outputs_shape)
               |> Cuda.Compiler.Utils.unwrap_pins

      {:ok, output}
    else
      _ -> {:error, :no_cuda_specified}
    end
  end
end
