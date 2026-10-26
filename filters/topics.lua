-- Вставляет в начало статьи блок «Уровень и связи»: уровень, статус, какие темы нужно
-- знать заранее и какие темы опираются на эту. Данные берутся из
-- _generated/topics.json, который пишет scripts/topics.py перед сборкой.

local topics = nil

local function load_topics()
  local path = pandoc.path.join({ quarto.project.directory, "_generated", "topics.json" })
  local f = io.open(path, "r")
  if not f then
    return {}
  end
  local data = pandoc.json.decode(f:read("a"))
  f:close()
  return data
end

local function links(ids)
  local items = {}
  for i, id in ipairs(ids) do
    local t = topics[id]
    if i > 1 then
      table.insert(items, pandoc.Str(", "))
    end
    if t.href then
      table.insert(items, pandoc.Link(t.title, t.href))
    else
      -- тема есть в topics.yml, но статья ещё не написана
      table.insert(items, pandoc.Str(t.title))
      table.insert(items, pandoc.Space())
      table.insert(items, pandoc.Emph("(в работе)"))
    end
  end
  return items
end

local function line(label, ids)
  local inlines = { pandoc.Strong(label), pandoc.Space() }
  if #ids == 0 then
    table.insert(inlines, pandoc.Emph("—"))
  else
    for _, x in ipairs(links(ids)) do
      table.insert(inlines, x)
    end
  end
  return pandoc.Para(inlines)
end

function Pandoc(doc)
  local id = doc.meta.topic and pandoc.utils.stringify(doc.meta.topic)
  if not id then
    return doc
  end
  topics = topics or load_topics()
  local t = topics[id]
  if not t then
    return doc
  end
  local box = pandoc.Div({
    pandoc.Header(2, "Уровень и связи"),
    pandoc.Para({
      pandoc.Strong("Уровень:"), pandoc.Space(), pandoc.Str(string.format("%d — %s", t.level, t.level_name)),
      pandoc.Str(";"), pandoc.Space(),
      pandoc.Strong("статус:"), pandoc.Space(), pandoc.Str(string.format("[%s] %s", t.status, t.status_name)),
    }),
    line("Нужно знать:", t.requires),
    line("Куда дальше:", t.required_by),
  }, pandoc.Attr("", { "callout-note" }, { collapse = "false" }))
  table.insert(doc.blocks, 1, box)
  return doc
end
