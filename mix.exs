defmodule Cuda.Mixfile do
  use Mix.Project

  def project do
    [app: :cuda,
     version: "0.1.0",
     elixir: "~> 1.4",
     build_embedded: Mix.env == :prod,
     start_permanent: Mix.env == :prod,
     compilers: [:port, :elixir, :app],
     deps: deps()]
  end

  def application do
    [#mod: {Cuda.App, []},
     extra_applications: [:logger]]
  end

  defp deps do
    []
  end
end

defmodule Mix.Tasks.Compile.Port do
  @cuda_version_file "/usr/local/cuda/version.txt"
  @cuda_version_re ~r/\s+(\d+\.\d+)/
  def run(_) do
    cuda = with true <- File.exists?(@cuda_version_file),
                {:ok, version} <- File.read(@cuda_version_file),
                [_, version] <- Regex.run(@cuda_version_re, version) do
      "cuda-#{version}"
    else
      _ -> "cuda"
    end
    opts = [stderr_to_stdout: true,
            env: [{"CUDA", cuda}]]
    if match? {:win32, _}, :os.type do
      {result, _error_code} = System.cmd("nmake", ["priv\\cuda_port.exe"], opts)
      Mix.shell.info result
    else
      {result, _error_code} = System.cmd("make", ["priv/cuda_port"], opts)
      Mix.shell.info result
    end
  end
end
