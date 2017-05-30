defmodule Cuda.Template.HelpersTest do
  use ExUnit.Case
  alias Cuda.Compiler.Context
  #import Kernel, except: ["@": 2]
  import Cuda.Template.Helpers, except: ["@": 2]

  describe "env/2" do
    test "Get environment variable from context" do
      ctx = %Context{env: %Cuda.Env{int_size: 16}}
      assert 16 == env(ctx, :int_size)
    end
  end

  describe "var/2" do
    test "gets variable from context" do
      ctx = %Context{assigns: %{vars: %{var: 16}}}
      assert 16 == var(ctx, :var)
    end

    test "gets variable from context path" do
      root = %Cuda.Graph{id: :root, assigns: %{vars: %{c: 30}}, nodes: [
        %Cuda.Graph{id: :b, assigns: %{vars: %{b: 20}}, nodes: [
          %Cuda.Graph.Node{id: :a, assigns: %{vars: %{a: 10}}}
        ]}
      ]}
      ctx = %Context{root: root, path: [:a, :b]}
      assert 10 = var(ctx, :a)
      assert 20 = var(ctx, :b)
      assert 30 = var(ctx, :c)
    end
  end
end
