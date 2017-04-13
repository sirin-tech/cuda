defmodule EnvTest do
  use ExUnit.Case
  alias Cuda.Env

  describe "create" do
    test "env map with default values" do
      default = Map.merge(%Env{}, Env.get_default())

      assert {:ok, default} == Env.create
    end
  end

  describe "load" do
    test "env map with test config values" do
      tmp_var = System.get_env("CUDA_ENV")
      System.put_env("CUDA_ENV", "values_for_cuda_testing")

      answer = Map.merge(%Env{}, Env.get_default())
      answer = :cuda
      |> Application.get_env(:values_for_cuda_testing)
      |> Enum.reduce(answer, fn ({key, val}, acc) -> Map.put(acc, key, val) end)

      {:ok, env} = Env.load()

      if tmp_var == nil do
        System.delete_env("CUDA_ENV")
      else
        System.put_env("CUDA_ENV", tmp_var)
      end

      assert answer == env
    end
  end

  describe "merge" do
    test "env map merges with test keywordlist" do
      kw = Application.get_env(:cuda, :values_for_cuda_testing)
      initial = Map.merge(%Env{}, Env.get_default())
      answer = Enum.reduce(kw, initial, fn ({key, val}, acc) -> Map.put(acc, key, val) end)

      assert {:ok, answer} == Env.merge(initial, kw)
    end

    test "get an error when try to load not permitted value" do
      assert {:error, _} = Env.merge(%Env{}, [float_size: 0])
    end
  end

  describe "validate" do
    test "get an error when try to pass wrong value" do
      assert {:error, _} = Env.Validation.validate(:float_size, 0)
      assert {:error, _} = Env.Validation.validate(:int_size, 0)
      assert {:error, _} = Env.Validation.validate(:memory_optimization, 0)
    end
  end
end
