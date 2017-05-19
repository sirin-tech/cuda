defmodule Cuda.Shared do
  use GenServer
  alias Cuda.Graph.Pin

  def start_link(opts \\ []) do
    {opts, args} = Keyword.split(opts, ~w(name)a)
    GenServer.start_link(__MODULE__, args, opts)
  end

  def init(opts) do
    with {:ok, cuda} <- Cuda.start_link() do
      st = %{cuda: cuda, extracts: %{}, ref: nil}
      opts |> Keyword.get(:vars, %{}) |> load_vars(st)
    end
  end

  def load(pid, vars) do
    GenServer.call(pid, {:load, vars})
  end

  def unload(pid) do
    GenServer.call(pid, :unload)
  end

  def handle(pid) do
    GenServer.call(pid, :handle)
  end

  def offsets(pid) do
    GenServer.call(pid, :offsets)
  end

  def extracts(pid) do
    GenServer.call(pid, :extracts)
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

  def handle_call(:offsets, _from, st) do
    offsets = st.extracts
              |> Enum.map(&collect_offsets/1)
              |> Enum.into(%{})
    {:reply, {:ok, offsets}, st}
  end

  def handle_call(:extarcts, _from, st) do
    extracts = st.extracts
               |> Enum.map(fn {k, {o, s, _}} -> {k, {o, s}} end)
               |> Enum.into(%{})
    {:reply, {:ok, extracts}, st}
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
      vars = st.extracts |> Enum.reduce(%{}, fn {name, {o, s, t}}, vars ->
        data = data |> :binary.part({o, s}) |> Pin.unpack(t)
        Map.put(vars, name, data)
      end)
      {:ok, vars}
    end
    {:reply, result, st}
  end

  defp collect_offsets({k, {o, _, t}}) when is_map(t) do
    {o, _} = Enum.reduce(t, {%{}, 0}, fn {name, type}, {map, offset} ->
      size = Pin.type_size(type)
      {Map.put(map, name, o + offset), offset + size}
    end)
    {k, o}
  end
  defp collect_offsets({k, {o, _, _}}) do
    {k, o}
  end

  defp load_vars(vars, st) do
    {extracts, bin} = Enum.reduce(vars, {%{}, <<>>}, fn
      {k, {type, value}}, {vars, bin} ->
        value = Pin.pack(value, type)
        offset = byte_size(bin)
        size = byte_size(value)
        {Map.put(vars, k, {offset, size, type}), bin <> value}
      _, error ->
        error
    end)
    if byte_size(bin) > 0 do
      unload = case st.ref do
        nil -> :ok
        ref -> Cuda.memory_unload(st.cuda, ref)
      end
      with :ok <- unload,
           {:ok, ref} <- Cuda.memory_load(st.cuda, bin) do
        {:ok, %{st | ref: ref, extracts: extracts}}
      end
    else
      {:ok, %{st | ref: nil, extracts: extracts}}
    end
  end
end
