defmodule Cuda.Test.CudaHelpers do
  alias Cuda.Compiler.Context

  def env(values \\ []) do
    {:ok, env} = Cuda.Env.create()
    env
    |> Map.merge(%{gpu_info: gpu_info()})
    |> Map.merge(values |> Enum.into(%{}))
  end

  def context(values \\ []) do
    values = values |> Enum.into(%{})
    %Context{env: env(Map.get(values, :env, [])), assigns: %{vars: %{}}}
    |> Map.merge(values, &context_merge/3)
  end

  @header_directives ~w(.version .target .address_size)
  def parse_ptx(ptx) do
    ptx
    |> String.split("\n")
    |> Enum.map(&String.trim/1)
    |> Enum.map(& String.replace(&1, ~r/\s+/, " "))
    |> Enum.map(& String.split(&1, " "))
    |> Enum.reject(& List.first(&1) in @header_directives)
    |> Enum.map(& Enum.join(&1, " "))
    |> Enum.join()
    |> String.split(";")
  end

  def parse_c(c) do
    c
    |> String.split("\n")
    |> Enum.map(&String.trim/1)
    |> Enum.map(& String.replace(&1, ~r/\s+/, " "))
    |> Enum.join()
    |> String.split(";")
  end

  def gpu_info() do
    [max_threads_per_block: 1024, max_block: {1024, 1024, 64},
     max_grid: {2147483647, 65535, 65535}, max_shared_memory_per_block: 49152,
     total_constant_memory: 65536, warp_size: 32, max_pitch: 2147483647,
     max_registers_per_block: 65536, clock_rate: 1006000, gpu_overlap: true,
     miltiprocessor_count: 2, kernel_exec_timeout: true, integrated: false,
     can_map_host_memory: true, compute_mode: :default, concurrent_kernels: true,
     ecc_enabled: false, pci_bus_id: 1, pci_device_id: 0, tcc_driver: false,
     memory_clock_rate: 2505000, global_memory_bus_width: 64, l2_cache_size: 524288,
     max_threads_per_multiprocessor: 2048, unified_arressing: true,
     compute_capability: {3, 5}, global_l1_cache_supported: false,
     glocal_l1_cache_supported: true, max_shared_memory_per_multiprocessor: 49152,
     max_registers_per_multiprocessor: 65536, managed_memory: true,
     multi_gpu_board: false, multi_gpu_board_group_id: 0,
     host_native_atomic_supported: false, single_to_double_precision_perf_ratio: 24,
     pageable_memory_access: false, concurrent_managed_access: false,
     compute_preemption_supported: false,
     can_use_host_pointer_for_registered_mem: false]
  end

  defp context_merge(:env, v1, v2), do: Map.merge(v1, v2)
  defp context_merge(:assigns, v1, v2), do: Map.merge(v1, v2)
  defp context_merge(_, _v1, v2), do: v2
end
