defmodule Cuda.Mixfile do
  use Mix.Project

  def project do
    [app: :cuda,
     version: "0.1.0",
     elixir: "~> 1.4",
     build_embedded: Mix.env == :prod,
     start_permanent: Mix.env == :prod,
     compilers: [:port, :elixir, :app],
     elixirc_paths: paths(),
     deps: deps(),
     aliases: aliases(),
     docs: docs()]
  end

  def application do
    [mod: {Cuda.App, []},
     extra_applications: [:logger]]
  end

  defp deps do
    [{:uuid, "~> 1.1"},
     # {:cpp_port, path: "../cpp_port"},
     {:credo, "~> 0.7", only: [:dev, :test]},
     {:ex_doc, "~> 0.15", only: :dev, runtime: false}]
  end

  defp aliases do
    [clean: ["clean.port", "clean"]]
  end

  defp docs do
    [main: "Cuda",
     #logo: "path/to/logo.png",
     extras: ["README.md"]]
  end

  defp paths do
    ["lib", Path.join(~w(test support))]
  end

  # defp cpp_ports do
  #   [[module: A,
  #     src: "src",
  #     target: "priv/cuda_driver_port_test",
  #     env: %{"CUDA" => "cuda-8.0"}]]
  # end
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
      {result, _error_code} = System.cmd("nmake", ["priv\\cuda_driver_port.exe"], opts)
      Mix.shell.info result
    else
      {result, _error_code} = System.cmd("make", ["priv/cuda_driver_port"], opts)
      Mix.shell.info result
    end
  end
end

defmodule Mix.Tasks.Clean.Port do
  def run(_) do
    opts = [stderr_to_stdout: true]
    if match? {:win32, _}, :os.type do
      {result, _error_code} = System.cmd("nmake", ["clean"], opts)
      Mix.shell.info result
    else
      {result, _error_code} = System.cmd("make", ["clean"], opts)
      Mix.shell.info result
    end
  end
end
