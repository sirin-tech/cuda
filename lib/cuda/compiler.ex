defmodule Cuda.Compiler do
  def compile(sources, opts) do
    tmp = Path.join(System.tmp_dir!, "CudaCompiler-#{UUID.uuid1()}")
    File.mkdir_p!(tmp)
    files = sources
            |> Enum.reduce([], fn
              {:ptx, src}, acc ->
                file = Path.join(tmp, "#{UUID.uuid1()}.ptx")
                File.write(file, src)
                [file | acc]
              {:c, src}, acc ->
                file = Path.join(tmp, "#{UUID.uuid1()}.ptx")
                File.write(file, src)
                [file | acc]
              _, acc ->
                acc
            end)
    nvcc = Keyword.get(opts, :nvcc, "/usr/local/cuda/bin/nvcc")
    cubin = Path.join(tmp, "#{UUID.uuid1}.cubin")
    args = ~w(-o #{cubin}) ++ files
    result = with {_, 0} <- System.cmd(nvcc, args, opts) do
      File.read(cubin)
    end
    File.rm_rf!(tmp)
    result
  end
end
