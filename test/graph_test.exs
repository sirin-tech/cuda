defmodule GraphTest do
  use ExUnit.Case
  alias Cuda.Graph
  alias Cuda.Graph.Node

  defmodule Double do
    use Node
    def __pins__(_, _) do
      [input(:input1, :i8), input(:input2, :i8),
       output(:output1, :i8), output(:output2, :i8)]
    end
    def __type__(_, _), do: :virtual
  end

  defmodule Single do
    use Node
    def __pins__(_, _) do
      [input(:input, :i8), output(:output, :i8)]
    end
    def __type__(_, _), do: :virtual
  end

  defmodule NoOut do
    use Node
    def __pins__(_, _) do
      [input(:input, :i8)]
    end
    def __type__(_, _), do: :virtual
  end

  defmodule DummyGraph do
    use Graph
    def __pins__(_, _) do
      [input(:input, :i8), output(:output, :i8)]
    end
    def __graph__(graph, _, _), do: graph
  end

  import Graph, except: [graph: 1, graph: 2]

  def eval_macro(str) do
    opts = [
      macros: [{Cuda.Graph, graph: 1, graph: 2} | __ENV__.macros]
    ]
    {result, _} = Code.eval_string(str, [], opts)
    result
  end

  describe "output/2" do
    test "rejects calls with wrong arguments" do
      assert_raise(CompileError, fn -> output(nil) end)
      assert_raise(CompileError, fn -> %Graph{} |> output(:test) end)
    end

    test "connects graph output to previous node" do
      assert %{links: [{:input, :output}]} = %Graph{} |> output()
    end

    test "connects to specified connector in previous node" do
      graph = %Graph{} |> run(Double, name: :a) |> output(:output2)
      assert [{:input, {:a, :input1}},
              {{:a, :output2}, :output}] = graph.links
    end

    test "connects to specified node and connector" do
      graph = %Graph{}
              |> run(Double, name: :a)
              |> run(Single, name: :b)
              |> output({:a, :output2})
      assert [{:input, {:a, :input1}},
              {{:a, :output1}, {:b, :input}},
              {{:a, :output2}, :output}] = graph.links
    end
  end

  describe "connect/3" do
    test "rejects calls with wrong arguments" do
      assert_raise(CompileError, fn -> connect(nil, :a) end)
      assert_raise(CompileError, fn -> %Graph{} |> connect(:a) end)
    end

    test "connects to previous node with one arg" do
      graph = %Graph{}
              |> run(Double, name: :a)
              |> run(Single, name: :b)
              |> connect(:a)
      assert [{:input, {:a, :input1}},
              {{:a, :output1}, {:b, :input}},
              {{:b, :output}, {:a, :input2}}] = graph.links
      graph = %Graph{}
              |> run(Double, name: :a, input: :input2)
              |> run(Single, name: :b)
              |> connect({:a, :input1})
      assert [{:input, {:a, :input2}},
              {{:a, :output1}, {:b, :input}},
              {{:b, :output}, {:a, :input1}}] = graph.links
    end

    test "connects src to dst with two args" do
      graph = %Graph{}
              |> run(Double, name: :a)
              |> run(Single, name: :b)
              |> connect(:a, :a)
      assert [{:input, {:a, :input1}},
              {{:a, :output1}, {:b, :input}},
              {{:a, :output2}, {:a, :input2}}] = graph.links
      graph = %Graph{}
              |> run(Double, name: :a)
              |> run(Double, name: :b, input: :input2)
              |> connect({:a, :output2}, {:b, :input1})
      assert [{:input, {:a, :input1}},
              {{:a, :output1}, {:b, :input2}},
              {{:a, :output2}, {:b, :input1}}] = graph.links
    end
  end

  describe "run/3" do
    test "rejects calls with wrong arguments" do
      assert_raise(CompileError, fn -> run(nil, Double) end)
      assert_raise(CompileError, fn -> %Graph{} |> run(Test) end)
    end

    test "connects to previous node" do
      graph = %Graph{}
              |> run(Single, name: :a)
      assert [%Node{id: :a}] = graph.nodes
      assert [{:input, {:a, :input}}] = graph.links
      graph = %Graph{}
              |> run(Single, name: :a)
              |> run(Single, name: :b)
      assert [%Node{id: :a}, %Node{id: :b}] = graph.nodes
      assert [{:input, {:a, :input}},
              {{:a, :output}, {:b, :input}}] = graph.links
    end

    test "raises error when there are no outputs in previous node" do
      assert_raise(CompileError, fn -> %Graph{} |> run(NoOut) |> run(Single) end)
    end

    test "connects to specified out in previous node" do
      graph = %Graph{}
              |> run(Double, name: :a)
              |> run(Single, name: :b, source: :output2)
      assert [{:input, {:a, :input1}},
              {{:a, :output2}, {:b, :input}}] = graph.links
    end

    test "connects to specified node and out" do
      graph = %Graph{}
              |> run(Double, name: :a)
              |> run(Single, name: :b)
              |> run(Single, name: :c, source: {:a, :output2})
      assert [{:input, {:a, :input1}},
              {{:a, :output1}, {:b, :input}},
              {{:a, :output2}, {:c, :input}}] = graph.links
    end

    test "connects specified connector as input" do
      graph = %Graph{}
              |> run(Double, name: :a, input: :input2)
      assert [{:input, {:a, :input2}}] = graph.links
    end
  end

  describe "validate!/1" do
    test "rejects graph with wrong input number" do
      assert_raise(CompileError, fn ->
        %Graph{} |> validate!
      end)
      assert_raise(CompileError, fn ->
        %Graph{} |> output() |> output() |> validate!
      end)
    end

    test "rejects graph with wrong output number" do
      assert_raise(CompileError, fn ->
        %Graph{} |> run(Single) |> validate!
      end)
      assert_raise(CompileError, fn ->
        %Graph{} |> run(Single) |> output() |> output() |> validate!
      end)
    end

    test "rejects graph with unconnected connectors" do
      assert_raise(CompileError, fn ->
        %Graph{} |> run(Double) |> output() |> validate!
      end)
    end
  end
end
