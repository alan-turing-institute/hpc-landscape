# Isambard Status

There is no function on Isambard Phase 2 which displays whether GPUs are available, but it is possible to inspect the nodes using `sinfo`.

> Running `squeue` or `sinfo` in a rapid loop (e.g. using `watch` or the `--iterate` flag with a short interval) floods the Slurm scheduler with queries. 
> **This can slow down job scheduling for every user on the system, not just your own jobs.**
> Do not run the inspection scripts twice in one minute.

The following script shows how many GPUs are available with the CPUs and Memory that would be provided when using the `--gpus` flag on `sbatch`.

```bash
sinfo --Format="nodehost:15,gres:25,gresused:40,cpusstate:15,memory:12,allocmem:12,statelong:12" \
  --partition=workq --states=mix,idle | \
  grep -v "mixed-\|idle-" | \
  awk '
  NR>1 {
    s = tolower($7)
    if (s != "mixed" && s != "idle") next
    split($2, gtot, ":"); split(gtot[2], t, "("); gpu_total = t[1]+0
    split($3, gused, ":"); split(gused[3], u, "("); gpu_used = u[1]+0
    split($4, cpu, "/")
    free_gpu = gpu_total - gpu_used
    if (free_gpu > 0) {
      mem_free = ($5 - $6) / 1024; mem_total = $5 / 1024
      gpu_frac = (gpu_total > 0) ? free_gpu / gpu_total : 0
      cpu_frac = (cpu[4] > 0) ? cpu[2] / cpu[4] : 0
      mem_frac = (mem_total > 0) ? mem_free / mem_total : 0
      avail = (gpu_frac < cpu_frac) ? gpu_frac : cpu_frac
      avail = (avail < mem_frac) ? avail : mem_frac
      if      (avail >= 1.0)  bin4++
      else if (avail >= 0.75) bin3++
      else if (avail >= 0.5)  bin2++
      else if (avail >= 0.25) bin1++
    }
  }
  END {
    printf "Node Availability\n"
    printf "Quarter Free        : %d\n", bin1+0
    printf "Half Free           : %d\n", bin2+0
    printf "Three-Quarters Free : %d\n", bin3+0
    printf "Completely Free     : %d\n", bin4+0
    printf "--------------------------\n"
    printf "Total GPU slot      : %d\n", bin1+2*bin2+3*bin3+4*bin4+0
  }'
```

This next script shows all the nodes with a GPU available.
It also displays how much memory and how many CPUs are available for each of these nodes.

```bash
sinfo --Format="nodehost:15,gres:25,gresused:40,cpusstate:15,memory:12,allocmem:12,statelong:12" \
  --partition=workq --states=mix,idle | \
  grep -v "mixed-\|idle-" | \
  awk '
  BEGIN { print "NODE\tSTATE\tGPU\tCPU\tMEM\tAVAIL" }
  NR>1 {
    s = tolower($7)
    if (s != "mixed" && s != "idle") next
    split($2, gtot, ":"); split(gtot[2], t, "("); gpu_total = t[1]+0
    split($3, gused, ":"); split(gused[3], u, "("); gpu_used = u[1]+0
    split($4, cpu, "/")
    free_gpu = gpu_total - gpu_used
    if (free_gpu > 0) {
      mem_free = ($5 - $6) / 1024; mem_total = $5 / 1024
      gpu_frac = (gpu_total > 0) ? free_gpu / gpu_total : 0
      cpu_frac = (cpu[4] > 0) ? cpu[2] / cpu[4] : 0
      mem_frac = (mem_total > 0) ? mem_free / mem_total : 0
      avail = (gpu_frac < cpu_frac) ? gpu_frac : cpu_frac
      avail = (avail < mem_frac) ? avail : mem_frac
      if (avail >= 0.25) count25++
      nodes++
      printf "%s\t%s\t%d/%d\t%d/%d\t%d/%d\t%.2f\n",
        $1, $7, free_gpu, gpu_total, cpu[2], cpu[4],
        int(mem_free), int(mem_total), avail
    }
  }
  END {
    printf "\n%d nodes with at least 1 GPU available\n", nodes+0
    printf "%d nodes with AVAIL >= 0.25\n", count25+0
  }' | column -t
```
It is possible for a GPU to be free, but it will not be allocated when calling `--gpus=1` because there is not sufficient memory or CPUs to fit the default allocation.
If this is the case, it is possible to reduce the allocation by using the `--mem-per-gpu` flag and the `--cpus-per-gpu` flag.
There is an issue with the `--mem` flag on Isambard, which is overwritten by the default `--mem-per-gpu` value.

