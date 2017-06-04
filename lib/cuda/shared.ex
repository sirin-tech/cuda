defmodule Cuda.Shared do
  use GenServer
  alias Cuda.Memory

  def start_link(opts \\ []) do
    {opts, args} = Keyword.split(opts, ~w(name)a)
    GenServer.start_link(__MODULE__, args, opts)
  end

  def init(opts) do
    with {:ok, cuda} <- Cuda.start_link() do
      st = %{cuda: cuda, memory: Keyword.get(opts, :memory), ref: nil}
      opts |> Keyword.get(:vars, %{}) |> load_vars(st)
    end
  end

  def load(pid, vars) do
    GenServer.call(pid, {:load, vars})
  end
  def load(pid, memory, vars) do
    GenServer.call(pid, {:load, memory, vars})
  end

  def unload(pid) do
    GenServer.call(pid, :unload)
  end

  def handle(pid) do
    GenServer.call(pid, :handle)
  end

  def memory(pid) do
    GenServer.call(pid, :memory)
  end

  def data(pid) do
    GenServer.call(pid, :data)
  end

  def share(pid) do
    GenServer.call(pid, :share)
  end

  def vars(pid) do
    GenServer.call(pid, :vars)
  end

  def handle_call({:load, vars}, _from, st) do
    with {:ok, st} <- load_vars(vars, st) do
      {:reply, {:ok, st.ref}, st}
    end
  end

  def handle_call({:load, memory, vars}, _from, st) do
    st = %{st | memory: memory}
    with {:ok, st} <- load_vars(vars, st) do
      {:reply, {:ok, st.ref}, st}
    end
  end

  def handle_call(:unload, _from, %{ref: ref} = st) when not is_nil(ref) do
    with :ok <- Cuda.memory_unload(st.cuda, st.ref) do
      {:reply, :ok, %{st | extracts: %{}, ref: nil}}
    else
      result -> {:reply, result, st}
    end
  end
  def handle_call(:unload, _from, st) do
    {:reply, :ok, st}
  end

  def handle_call(:handle, _from, st) do
    {:reply, {:ok, st.ref}, st}
  end

  def handle_call(:memory, _from, st) do
    {:reply, {:ok, st.memory}, st}
  end

  def handle_call(:data, _from, %{ref: ref} = st) when not is_nil(ref) do
    result = Cuda.memory_read(st.cuda, ref)
    {:reply, result, st}
  end
  def handle_call(:data, _from, st) do
    {:reply, {:ok, nil}, st}
  end

  def handle_call(:share, _from, st) do
    result = Cuda.memory_share(st.cuda, st.ref)
    {:reply, result, st}
  end

  def handle_call(:vars, _from, %{ref: ref} = st) when not is_nil(ref) do
    result = with {:ok, data} <- Cuda.memory_read(st.cuda, ref) do
      {:ok, Memory.unpack(data, st.memory)}
    end
    {:reply, result, st}
  end

  defp load_vars(_, %{memory: nil} = st), do: {:ok, st}
  defp load_vars(vars, %{memory: memory} = st) do
    #IO.inspect({vars, memory})
    bin = Memory.pack(vars, memory)
    #IO.inspect({memory, vars, (for <<x::float-little-32 <- bin>>, do: x)})
    if byte_size(bin) > 0 do
      unload = case st.ref do
        nil -> :ok
        ref -> Cuda.memory_unload(st.cuda, ref)
      end
      with :ok <- unload,
           {:ok, ref} <- Cuda.memory_load(st.cuda, bin) do
        {:ok, %{st | ref: ref}}
      end
    else
      {:ok, %{st | ref: nil}}
    end
  end
end
