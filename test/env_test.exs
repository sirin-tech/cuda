defmodule Cuda.EnvTest do
  use ExUnit.Case
  import Cuda.Env
  import Cuda.Env.Validation

  describe "create" do
    test "env map with default values" do
      default = Map.merge(%Cuda.Env{}, get_default())

      assert {:ok, default} == create()
    end
  end

  describe "load" do
    setup do
      tmp_var = System.get_env("CUDA_ENV")
      System.put_env("CUDA_ENV", "values_for_cuda_testing")

      on_exit fn ->
        if tmp_var == nil do
          System.delete_env("CUDA_ENV")
        else
          System.put_env("CUDA_ENV", tmp_var)
        end
      end
    end

    test "env map with test config values" do
      answer = Map.merge(%Cuda.Env{}, get_default())
      answer = :cuda
               |> Application.get_env(:values_for_cuda_testing)
               |> Enum.reduce(answer, fn ({key, val}, acc) -> Map.put(acc, key, val) end)

      {:ok, env} = load()

      assert answer == env
    end
  end

  describe "merge/2" do
    test "env map merges with test keywordlist" do
      kw = Application.get_env(:cuda, :values_for_cuda_testing)
      initial = Map.merge(%Cuda.Env{}, get_default())
      answer = Enum.reduce(kw, initial, fn ({key, val}, acc) -> Map.put(acc, key, val) end)

      assert {:ok, answer} == merge(initial, kw)
    end

    test "get an error when try to load not permitted value" do
      assert {:error, _} = merge(%Cuda.Env{}, [float_size: 0])
    end
  end

  describe "validate/2" do
    test "get an error when try to pass wrong value" do
      assert {:error, _} = validate(:float_size, 0)
      assert {:error, _} = validate(:int_size, 0)
      assert {:error, _} = validate(:optimize, 0)
    end
  end
end
