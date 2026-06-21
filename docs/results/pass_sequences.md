# Лучшие последовательности пассов (cbench)

Для каждого из 23 cbench-бенчмарков — последовательность пассов, найденная
`segment_tree_beam` (лучший префикс по `.text`): длина, сэкономленные байты
`.text`, сэкономленные машинные инструкции и сама последовательность.

| benchmark | len | −.text Б | −instr | последовательность пассов |
|---|--:|--:|--:|---|
| adpcm | 2 | 363 | 48 | `early-cse sroa` |
| bitcount | 6 | 446 | 51 | `nary-reassociate early-cse dse newgvn correlated-propagation sroa` |
| blowfish | 16 | 5213 | 1136 | `gvn-sink gvn-sink sroa attributor attributor speculative-execution adce alignment-from-assumptions attributor ipsccp attributor attributor attributor slp-vectorizer attributor newgvn` |
| bzip2 | 16 | 23579 | 4000 | `nary-reassociate loop-rotate mem2reg gvn-sink gvn-sink simplifycfg newgvn gvn-hoist simplifycfg loop-versioning-licm iroutliner sroa newgvn mergefunc licm memcpyopt` |
| crc32 | 8 | 60 | 0 | `sroa ipsccp dse separate-const-offset-from-gep mergefunc gvn-sink alignment-from-assumptions loop-rotate` |
| dijkstra | 1 | 187 | 24 | `mem2reg` |
| ghostscript | 16 | 298769 | 46973 | `instsimplify memcpyopt constmerge indvars mldst-motion newgvn simplifycfg early-cse mem2reg instcombine jump-threading ipsccp gvn-sink gvn-hoist attributor slp-vectorizer` |
| gsm | 16 | 14491 | 2752 | `lower-expect mem2reg separate-const-offset-from-gep iroutliner attributor early-cse jump-threading correlated-propagation gvn ipsccp tailcallelim dse gvn slp-vectorizer loop-rotate simplifycfg` |
| ispell | 16 | 13354 | 1612 | `attributor jump-threading newgvn newgvn gvn-sink mem2reg globaldce attributor attributor correlated-propagation loop-deletion constmerge attributor mldst-motion gvn-hoist speculative-execution` |
| jpeg-c | 16 | 43486 | 7250 | `newgvn gvn-sink simplifycfg sroa gvn-sink mergefunc lcssa gvn-sink ipsccp jump-threading gvn-sink gvn-hoist newgvn jump-threading mldst-motion lcssa` |
| jpeg-d | 16 | 41398 | 7021 | `mergefunc lcssa sroa simplifycfg jump-threading gvn slp-vectorizer instcombine loop-interchange mldst-motion gvn-sink simplifycfg gvn-hoist gvn-sink dse attributor` |
| lame | 16 | 43096 | 7151 | `function-attrs newgvn simplifycfg newgvn dse simplifycfg newgvn dse mem2reg attributor sroa gvn-sink aggressive-instcombine dfa-jump-threading div-rem-pairs jump-threading` |
| patricia | 14 | 1694 | 344 | `simplifycfg div-rem-pairs globalopt gvn simplifycfg sroa globalopt gvn simplifycfg simple-loop-unswitch globalopt gvn simplifycfg simple-loop-unswitch` |
| qsort | 3 | 228 | 12 | `sroa loop-load-elim loop-unroll-and-jam` |
| rijndael | 5 | 2297 | 238 | `loop-bound-split loop-load-elim mergeicmps sroa loop-bound-split` |
| sha | 3 | 538 | 61 | `mem2reg loop-vectorize sccp` |
| stringsearch | 9 | 1035 | 114 | `bdce simplifycfg newgvn mem2reg gvn-hoist newgvn mem2reg lower-constant-intrinsics loop-unroll-and-jam` |
| stringsearch2 | 2 | 443 | 57 | `mem2reg jump-threading` |
| susan | 16 | 14674 | 3294 | `loop-load-elim mergereturn mergereturn sroa iroutliner adce instcombine iroutliner instcombine globaldce iroutliner adce tailcallelim attributor mergereturn loop-simplifycfg` |
| tiff2bw | 16 | 41215 | 7321 | `gvn instcombine gvn-sink gvn-hoist sroa jump-threading gvn instcombine newgvn dse simple-loop-unswitch gvn-sink iroutliner jump-threading mergefunc constmerge` |
| tiff2rgba | 16 | 40627 | 7366 | `jump-threading sroa gvn-sink newgvn jump-threading licm iroutliner jump-threading iroutliner ipsccp gvn-hoist simple-loop-unswitch gvn-sink aggressive-instcombine deadargelim mergefunc` |
| tiffdither | 16 | 40631 | 7433 | `argpromotion newgvn sroa jump-threading newgvn attributor simplifycfg loop-flatten mergefunc gvn-sink gvn-hoist constmerge gvn-sink aggressive-instcombine iroutliner iroutliner` |
| tiffmedian | 16 | 43781 | 8029 | `attributor gvn-sink sroa newgvn simplifycfg loop-versioning-licm jump-threading gvn-sink simple-loop-unswitch aggressive-instcombine gvn-hoist mergefunc iroutliner iroutliner constmerge gvn-sink` |
