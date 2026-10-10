## Pre-bake of the road profiles (docs/ROAD_NETWORK.md section 6 and risk R3,
## owner RS): every drawn RoadGraph edge solved once by RoadBuild on the base
## terrain (the shipped region packs and global layer, no HD packs, shape
## quality 2, every tile fully resident), written per AREA_DEG area to
## RoadBuild.BAKE_DIR/<area>.bin with an index (RoadBuild.BAKE_PATH). Each
## area carries its key (RoadBuild.area_key: the solver, the world rasters and
## the graph, packs, airports, hulls, stamps and city water within the area's
## box), each record the edge's rec_sig. At runtime RoadBuild.profile_edge opens
## the file of an edge's area on first use and decodes it instead of solving
## while both match; HD pack edges and edges the window could not hold fall back
## to the live solver and the user:// disk cache.
## Areas are independent: tools/bake_road_profiles_par.py runs several of
## these in parallel over the stale areas only and merges the index.
## Run (headless; the world is about 72 min of CPU in one process):
##   godot --headless --path . -s res://tools/bake_road_profiles.gd -- [fixture=1] [graph=<res path>]
##     [out=<dir>] [areas=all|<name>,<name>|@<list file>] [index=1|0] [keys=<json path>]
## fixture=1: the RGR1 fixture to tests/fixtures/roads/roadprof_fixture/.
## areas: area names (RoadBuild.area_name, "n30e110"), default all.
## index=0: do not touch index.json (parallel runs: the driver merges it).
## keys=<path>: bake nothing; write every area's current key, parts and edge
##   count to <path> (JSON) and quit.
## Every baked area prints one line "AREA {json}" (name, key, parts, records,
## bytes, km, skipped, ferries, ms) for the driver.
extends SceneTree

const FIXTURE_GRAPH := "res://tests/fixtures/roads/roadgraph_fixture.bin"
const FIXTURE_OUT := "res://tests/fixtures/roads/roadprof_fixture"

func _initialize() -> void:
	var args := {}
	for a in OS.get_cmdline_user_args():
		var kv := a.split("=", true, 1)
		args[kv[0]] = kv[1] if kv.size() > 1 else ""
	var fixture := args.has("fixture")
	var graph: String = args.get("graph", FIXTURE_GRAPH if fixture else RoadGraph.PATH)
	var out: String = args.get("out", FIXTURE_OUT if fixture else RoadBuild.BAKE_DIR)
	if out.begins_with("res://"):
		out = ProjectSettings.globalize_path(out)
	var t0 := Time.get_ticks_msec()
	Heightfield.load_data()
	Airports.load_db()
	Rivers.load_data()
	TerrainOverlay.init()
	Heightfield.quality = 2
	DetailTiles.use_hd = false # the base signature: HD pack edges solve live
	DetailTiles.init()
	LakeField.init()
	RoadGraph.use_file(graph)
	if not RoadGraph.init():
		print("bake_road_profiles: no graph at ", graph)
		quit(1)
		return
	RoadBuild.disk_cache = false
	RoadBuild.use_bake("")
	# the tiles of every area (an edge lives in the tile of its node a)
	var by_area := {}
	for k in RoadGraph.all_tiles():
		var a := RoadBuild.area_of(k)
		var ks: PackedInt32Array = by_area.get(a, PackedInt32Array())
		ks.append(k)
		by_area[a] = ks # (a packed array is a value: store it back)
	if args.has("keys"):
		_keys(by_area, String(args.keys), t0)
		return
	var want: Array = []
	var spec: String = args.get("areas", "all")
	if spec.begins_with("@"):
		spec = FileAccess.get_file_as_string(spec.substr(1)).replace("\r", "").replace("\n", ",")
	if spec == "all" or spec == "":
		want = by_area.keys()
	else:
		for n in spec.split(",", false):
			var a := RoadBuild.area_from_name(n.strip_edges())
			if not by_area.has(a):
				print("bake_road_profiles: no edges in area ", n)
				continue
			want.append(a)
	want.sort()
	DirAccess.make_dir_recursive_absolute(out)
	var results := {}
	var total_b := 0
	var total_r := 0
	for a: Vector2i in want:
		var r := _bake_area(a, by_area[a], out, fixture)
		if r.is_empty():
			quit(1)
			return
		results[RoadBuild.area_name(a)] = r
		total_b += int(r.bytes)
		total_r += int(r.records)
	if args.get("index", "1") != "0":
		_write_index(out, results, spec == "all" or spec == "", by_area)
	print("bake_road_profiles: %d areas, %d profiles, %d bytes, %d ms -> %s" % [want.size(), total_r, total_b, Time.get_ticks_msec() - t0, out])
	DetailTiles.clear()
	TerrainOverlay.shutdown()
	quit(0)

## Solves and writes one area; its result line (also printed as "AREA {json}").
func _bake_area(a: Vector2i, tiles: PackedInt32Array, out: String, fixture: bool) -> Dictionary:
	var t0 := Time.get_ticks_msec()
	var recs: Array = []
	var km := 0.0
	var skipped := 0
	var ferries := 0
	var sorted := tiles.duplicate()
	sorted.sort()
	for k in sorted:
		for e: Dictionary in RoadGraph.tile(k).edges:
			if (int(e.flags) & RoadGraph.F_FERRY) != 0:
				ferries += 1
				continue
			if not RoadBuild.load_terrain_for(e):
				skipped += 1 # longer than the detail window holds: solved live in the game
				continue
			var pr := RoadBuild._solve_edge(e)
			if pr.is_empty():
				continue
			var id: Vector2i = e.id
			recs.append([id.x, id.y, RoadBuild.rec_sig(e, RoadBuild.edge_sig(e)), RoadBuild.encode_profile(pr)])
			km += (pr.s as PackedFloat32Array)[-1] / 1000.0
	var name := RoadBuild.area_name(a)
	var bytes := RoadBuild.write_area(out.path_join(name + ".bin"), a, recs, fixture)
	if bytes < 0:
		print("bake_road_profiles: cannot write ", out.path_join(name + ".bin"))
		return {}
	var ak := RoadBuild.area_key(a)
	var r := { "name": name, "key": int(ak[0]), "parts": Array(ak[1]), "records": recs.size(), "bytes": bytes,
		"km": snappedf(km, 0.1), "skipped": skipped, "ferries": ferries, "ms": Time.get_ticks_msec() - t0 }
	print("AREA ", JSON.stringify(r))
	return r

## keys=<path>: every area's current key (the driver compares them with the
## index to find the stale areas).
func _keys(by_area: Dictionary, path: String, t0: int) -> void:
	var areas := {}
	var ks: Array = by_area.keys()
	ks.sort()
	for a: Vector2i in ks:
		var ak := RoadBuild.area_key(a)
		var ne := 0
		for k in by_area[a]:
			ne += RoadGraph.tile_info(k)[2] if RoadGraph.source == RoadGraph.SRC_FILE else (RoadGraph.tile(k).edges as Array).size()
		areas[RoadBuild.area_name(a)] = { "key": int(ak[0]), "parts": Array(ak[1]), "edges": ne }
	var doc := { "version": RoadBuild.BAKE_VERSION, "area_deg": RoadBuild.AREA_DEG, "margin_deg": RoadBuild.AREA_MARGIN,
		"global": RoadBuild.global_key(), "parts": RoadBuild.AREA_PARTS, "areas": areas }
	var f := FileAccess.open(path, FileAccess.WRITE)
	if f == null:
		print("bake_road_profiles: cannot write ", path)
		quit(1)
		return
	f.store_string(JSON.stringify(doc, "\t", true).replace("\n", "\r\n") + "\r\n")
	f.close()
	print("bake_road_profiles: keys of %d areas, %d ms -> %s" % [areas.size(), Time.get_ticks_msec() - t0, path])
	quit(0)

## index.json in out: the existing entries with the baked areas replaced (all:
## the areas without edges dropped, their files deleted).
func _write_index(out: String, results: Dictionary, all: bool, by_area: Dictionary) -> void:
	var path := out.path_join("index.json")
	var old: Variant = JSON.parse_string(FileAccess.get_file_as_string(path)) if FileAccess.file_exists(path) else null
	var areas: Dictionary = {}
	if old is Dictionary and int(old.get("version", 0)) == RoadBuild.BAKE_VERSION:
		areas = (old as Dictionary).get("areas", {})
	for n: String in results:
		var r: Dictionary = (results[n] as Dictionary).duplicate()
		r.erase("name")
		areas[n] = r
	var live := {}
	for a: Vector2i in by_area:
		live[RoadBuild.area_name(a)] = true
	for n: String in areas.keys():
		if not live.has(n) or (all and not results.has(n)):
			areas.erase(n)
			if FileAccess.file_exists(out.path_join(n + ".bin")):
				DirAccess.remove_absolute(out.path_join(n + ".bin"))
	var doc := { "version": RoadBuild.BAKE_VERSION, "area_deg": RoadBuild.AREA_DEG, "margin_deg": RoadBuild.AREA_MARGIN,
		"global": RoadBuild.global_key(), "parts": RoadBuild.AREA_PARTS, "areas": areas }
	var f := FileAccess.open(path, FileAccess.WRITE)
	f.store_string(JSON.stringify(doc, "\t", true).replace("\n", "\r\n") + "\r\n")
	f.close()
