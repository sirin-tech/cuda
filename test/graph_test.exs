defmodule GraphTest do
  use ExUnit.Case
  alias Cuda.Graph
  alias Cuda.Graph.Node
  alias Cuda.Graph.Connector
  import Graph, except: [graph: 1, graph: 2]

  defmodule Double do
    def connectors(_, _) do
      [%Connector{type: :input, id: :input1},
       %Connector{type: :input, id: :input2},
       %Connector{type: :output, id: :output1},
       %Connector{type: :output, id: :output2}]
    end
  end

  defmodule Single do
    def connectors(_, _) do
      [%Connector{type: :input, id: :input},
       %Connector{type: :output, id: :output}]
    end
  end

  defmodule NoOut do
    def connectors(_, _) do
      [%Connector{type: :input, id: :input}]
    end
  end

  def eval_macro(str) do
    opts = [
      macros: [{Cuda.Graph, graph: 1, graph: 2} | __ENV__.macros]
    ]
    {result, _} = Code.eval_string(str, [], opts)
    result
  end

  describe "graph/2" do
    test "rejects usage without do block" do
      assert_raise(CompileError, fn -> eval_macro(~s{graph()}) end)
      assert_raise(CompileError, fn -> eval_macro(~s{graph("test")}) end)
    end

    test "imports helpers" do
      assert %Graph{} = eval_macro("""
        graph do
          input()
          |> run(GraphTest.Double, name: :a, input: :input1)
          |> run(GraphTest.Single, name: :b, source: :output1)
          |> output()
          |> connect({:a, :output2}, {:a, :input2})
        end
      """)
    end

    test "validates graph" do
      assert_raise(CompileError, fn -> eval_macro("graph do\nend") end)
    end
  end

  describe "input/0" do
    test "produces empty graph" do
      assert %Graph{} = input()
    end
  end

  describe "output/2" do
    test "rejects calls with wrong arguments" do
      assert_raise(CompileError, fn -> output(nil) end)
      assert_raise(CompileError, fn -> input() |> output(:test) end)
    end

    test "connects graph output to previous node" do
      assert %{connections: [{:input, :output}]} = input() |> output()
    end

    test "connects to specified connector in previous node" do
      graph = input() |> run(Double, name: :a) |> output(:output2)
      assert [{:input, {:a, :input1}},
              {{:a, :output2}, :output}] = graph.connections
    end

    test "connects to specified node and connector" do
      graph = input()
              |> run(Double, name: :a)
              |> run(Single, name: :b)
              |> output({:a, :output2})
      assert [{:input, {:a, :input1}},
              {{:a, :output1}, {:b, :input}},
              {{:a, :output2}, :output}] = graph.connections
    end
  end

  describe "connect/3" do
    test "rejects calls with wrong arguments" do
      assert_raise(CompileError, fn -> connect(nil, :a) end)
      assert_raise(CompileError, fn -> input() |> connect(:a) end)
    end

    test "connects to previous node with one arg" do
      graph = input()
              |> run(Double, name: :a)
              |> run(Single, name: :b)
              |> connect(:a)
      assert [{:input, {:a, :input1}},
              {{:a, :output1}, {:b, :input}},
              {{:b, :output}, {:a, :input2}}] = graph.connections
      graph = input()
              |> run(Double, name: :a, input: :input2)
              |> run(Single, name: :b)
              |> connect({:a, :input1})
      assert [{:input, {:a, :input2}},
              {{:a, :output1}, {:b, :input}},
              {{:b, :output}, {:a, :input1}}] = graph.connections
    end

    test "connects src to dst with two args" do
      graph = input()
              |> run(Double, name: :a)
              |> run(Single, name: :b)
              |> connect(:a, :a)
      assert [{:input, {:a, :input1}},
              {{:a, :output1}, {:b, :input}},
              {{:a, :output2}, {:a, :input2}}] = graph.connections
      graph = input()
              |> run(Double, name: :a)
              |> run(Double, name: :b, input: :input2)
              |> connect({:a, :output2}, {:b, :input1})
      assert [{:input, {:a, :input1}},
              {{:a, :output1}, {:b, :input2}},
              {{:a, :output2}, {:b, :input1}}] = graph.connections
    end
  end

  describe "run/3" do
    test "rejects calls with wrong arguments" do
      assert_raise(CompileError, fn -> run(nil, Double) end)
      assert_raise(CompileError, fn -> input() |> run(Test) end)
    end

    test "connects to previous node" do
      graph = input()
              |> run(Single, name: :a)
      assert [%Node{id: :a}] = graph.nodes
      assert [{:input, {:a, :input}}] = graph.connections
      graph = input()
              |> run(Single, name: :a)
              |> run(Single, name: :b)
      assert [%Node{id: :a}, %Node{id: :b}] = graph.nodes
      assert [{:input, {:a, :input}},
              {{:a, :output}, {:b, :input}}] = graph.connections
    end

    test "raises error when there are no outputs in previous node" do
      assert_raise(CompileError, fn -> input() |> run(NoOut) |> run(Single) end)
    end

    test "connects to specified out in previous node" do
      graph = input()
              |> run(Double, name: :a)
              |> run(Single, name: :b, source: :output2)
      assert [{:input, {:a, :input1}},
              {{:a, :output2}, {:b, :input}}] = graph.connections
    end

    test "connects to specified node and out" do
      graph = input()
              |> run(Double, name: :a)
              |> run(Single, name: :b)
              |> run(Single, name: :c, source: {:a, :output2})
      assert [{:input, {:a, :input1}},
              {{:a, :output1}, {:b, :input}},
              {{:a, :output2}, {:c, :input}}] = graph.connections
    end

    test "connects specified connector as input" do
      graph = input()
              |> run(Double, name: :a, input: :input2)
      assert [{:input, {:a, :input2}}] = graph.connections
    end
  end

  describe "validate!/1" do
    test "rejects graph with wrong input number" do
      assert_raise(CompileError, fn ->
        input() |> validate!
      end)
      assert_raise(CompileError, fn ->
        input() |> output() |> output() |> validate!
      end)
    end

    test "rejects graph with wrong output number" do
      assert_raise(CompileError, fn ->
        input() |> run(Single) |> validate!
      end)
      assert_raise(CompileError, fn ->
        input() |> run(Single) |> output() |> output() |> validate!
      end)
    end

    test "rejects graph with unconnected connectors" do
      assert_raise(CompileError, fn ->
        input() |> run(Double) |> output() |> validate!
      end)
    end
  end
end
