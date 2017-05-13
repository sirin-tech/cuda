defmodule Cuda.Compiler do
  def compile(sources, opts \\ []) do
    tmp = Path.join(System.tmp_dir!, "CudaCompiler-#{UUID.uuid1()}")
    File.mkdir_p!(tmp)
    nvcc = Keyword.get(opts, :nvcc, "/usr/local/cuda/bin/nvcc")
    cubin = Path.join(tmp, "#{UUID.uuid1}.cubin")
    files = sources
            |> Enum.reduce([], fn
              {:ptx, src}, acc ->
                id = UUID.uuid1()
                file = Path.join(tmp, "#{id}.ptx")
                :ok  = File.write(file, src)
                [file | acc]
              {:c, src}, acc ->
                file = Path.join(tmp, "#{UUID.uuid1()}.cu")
                :ok  = File.write(file, src)
                [file | acc]
              _, acc ->
                acc
            end)
    args = ~w(-dlink --cubin -gencode arch=compute_30,code=sm_30 -o #{cubin}) ++ files
    result = with {_, 0} <- System.cmd(nvcc, args, opts) do
      File.read(cubin)
    end
    File.rm_rf!(tmp)
    result
  end
end
