defmodule Cuda.TestMemory do
  use ExUnit.Case

  describe "shared memory" do
    setup do
      {:ok, cuda1} = Cuda.start_link()
      {:ok, cuda2} = Cuda.start_link()
      [cuda1: cuda1, cuda2: cuda2]
    end

    test "shares memory", ctx do
      {:ok, a} = Cuda.memory_load(ctx[:cuda1], <<1, 2, 3, 4>>)
      {:ok, s} = Cuda.memory_share(ctx[:cuda1], a)
      {:ok, b} = Cuda.memory_load(ctx[:cuda2], s)
      {:ok, x} = Cuda.memory_read(ctx[:cuda2], b)
      assert x == <<1, 2, 3, 4>>
    end
  end
end
