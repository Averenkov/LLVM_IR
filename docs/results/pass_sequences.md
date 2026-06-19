# Лучшие последовательности пассов (cbench)

Для каждого benchmark-а и каждой эвристики - последовательность `best_passes` (лучший префикс по `.text`), длина и сэкономленные байты.

## adpcm

| эвристика | len | −байт | последовательность пассов |
|---|--:|--:|---|
| measured_superpath | 4 | 363 | `early-cse nary-reassociate speculative-execution sroa` |
| strong_links | 2 | 363 | `early-cse sroa` |
| segment_tree | 7 | 363 | `argpromotion constraint-elimination early-cse argpromotion constraint-elimination early-cse sroa` |
| flow_paths | 4 | 363 | `early-cse nary-reassociate speculative-execution sroa` |
| cycle_breaking | 8 | 363 | `loop-bound-split argpromotion lower-expect constraint-elimination early-cse nary-reassociate speculative-execution sroa` |
| bucket_dag_teacher | 2 | 363 | `early-cse sroa` |
| random_walk_1000 | 7 | 363 | `loop-bound-split argpromotion constraint-elimination early-cse nary-reassociate speculative-execution sroa` |

## bitcount

| эвристика | len | −байт | последовательность пассов |
|---|--:|--:|---|
| measured_superpath | 6 | 446 | `nary-reassociate early-cse dse newgvn correlated-propagation sroa` |
| strong_links | 3 | 430 | `early-cse dse sroa` |
| segment_tree | 6 | 446 | `alignment-from-assumptions dse newgvn attributor sroa correlated-propagation` |
| flow_paths | 2 | 414 | `loop-idiom sroa` |
| cycle_breaking | 2 | 414 | `loop-idiom sroa` |
| bucket_dag_teacher | 1 | 414 | `sroa` |
| random_walk_1000 | 2 | 414 | `loop-idiom sroa` |

## blowfish

| эвристика | len | −байт | последовательность пассов |
|---|--:|--:|---|
| measured_superpath | 9 | 5141 | `gvn-sink sroa alignment-from-assumptions instsimplify attributor mem2reg loop-interchange newgvn attributor` |
| strong_links | 3 | 5043 | `gvn-sink newgvn sroa` |
| segment_tree | 16 | 5197 | `gvn-sink sroa loop-vectorize loop-vectorize gvn-sink sroa attributor mem2reg ipsccp newgvn gvn-sink sroa inferattrs slp-vectorizer loop-vectorize attributor` |
| flow_paths | 4 | 4635 | `gvn-sink sroa slp-vectorizer adce` |
| cycle_breaking | 8 | 4795 | `loop-vectorize gvn-sink sroa inferattrs slp-vectorizer alignment-from-assumptions adce instsimplify` |
| bucket_dag_teacher | 4 | 4795 | `gvn-sink sroa slp-vectorizer instsimplify` |
| random_walk_1000 | 5 | 717 | `attributor speculative-execution ipsccp loop-interchange newgvn` |

## bzip2

| эвристика | len | −байт | последовательность пассов |
|---|--:|--:|---|
| measured_superpath | 19 | 21683 | `loop-bound-split newgvn mem2reg callsite-splitting inferattrs mem2reg loop-rotate memcpyopt jump-threading loop-versioning-licm reassociate mem2reg hotcoldsplit simplifycfg newgvn mem2reg vector-combine sroa newgvn` |
| strong_links | 14 | 22215 | `gvn-hoist ipsccp loop-simplifycfg early-cse mldst-motion mergefunc jump-threading sroa aggressive-instcombine simplifycfg gvn dse constmerge licm` |
| segment_tree | 9 | 21285 | `loop-bound-split newgvn mem2reg correlated-propagation jump-threading jump-threading loop-instsimplify sroa gvn-hoist` |
| flow_paths | 5 | 21101 | `slp-vectorizer loop-bound-split newgvn jump-threading mem2reg` |
| cycle_breaking | 9 | 21080 | `newgvn partially-inline-libcalls constmerge loop-simplify libcalls-shrinkwrap loop-versioning jump-threading loop-versioning-licm mem2reg` |
| bucket_dag_teacher | 4 | 20341 | `newgvn mem2reg simplifycfg loop-unroll-and-jam` |
| random_walk_1000 | 6 | 19101 | `loop-flatten gvn-hoist aggressive-instcombine slp-vectorizer licm sroa` |

## crc32

| эвристика | len | −байт | последовательность пассов |
|---|--:|--:|---|
| measured_superpath | 4 | 60 | `sroa ipsccp alignment-from-assumptions loop-rotate` |
| strong_links | 2 | 56 | `dse sroa` |
| segment_tree | 5 | 60 | `dse sroa ipsccp alignment-from-assumptions loop-rotate` |
| flow_paths | 1 | 56 | `sroa` |
| cycle_breaking | 8 | 60 | `sroa ipsccp dse separate-const-offset-from-gep mergefunc gvn-sink alignment-from-assumptions loop-rotate` |
| bucket_dag_teacher | 1 | 56 | `sroa` |
| random_walk_1000 | 1 | 56 | `sroa` |

## dijkstra

| эвристика | len | −байт | последовательность пассов |
|---|--:|--:|---|
| measured_superpath | 3 | 187 | `aggressive-instcombine adce mem2reg` |
| strong_links | 1 | 187 | `mem2reg` |
| segment_tree | 3 | 187 | `aggressive-instcombine adce mem2reg` |
| flow_paths | 2 | 187 | `adce mem2reg` |
| cycle_breaking | 3 | 187 | `aggressive-instcombine adce mem2reg` |
| bucket_dag_teacher | 1 | 187 | `mem2reg` |
| random_walk_1000 | 3 | 187 | `aggressive-instcombine adce mem2reg` |

## ghostscript

| эвристика | len | −байт | последовательность пассов |
|---|--:|--:|---|
| measured_superpath | 6 | 286144 | `loop-fusion argpromotion sroa jump-threading gvn attributor` |
| strong_links | 16 | 298769 | `instsimplify memcpyopt constmerge indvars mldst-motion newgvn simplifycfg early-cse mem2reg instcombine jump-threading ipsccp gvn-sink gvn-hoist attributor slp-vectorizer` |
| segment_tree | 16 | 291718 | `instcombine argpromotion sroa jump-threading newgvn aggressive-instcombine sroa jump-threading loop-instsimplify attributor sroa jump-threading early-cse simplifycfg attributor separate-const-offset-from-gep` |
| flow_paths | 5 | 282927 | `lcssa mem2reg speculative-execution newgvn jump-threading` |
| cycle_breaking | 12 | 280556 | `loop-deletion early-cse mergefunc loop-bound-split instsimplify loop-versioning-licm loop-instsimplify slp-vectorizer sroa jump-threading loop-load-elim aggressive-instcombine` |
| bucket_dag_teacher | 12 | 270335 | `mem2reg newgvn instsimplify loop-simplifycfg alignment-from-assumptions instcombine slp-vectorizer dfa-jump-threading loop-fusion attributor loop-load-elim globaldce` |
| random_walk_1000 | 11 | 286521 | `loop-sink constmerge mem2reg gvn callsite-splitting gvn-hoist memcpyopt dfa-jump-threading loop-unroll-and-jam sroa jump-threading` |

## gsm

| эвристика | len | −байт | последовательность пассов |
|---|--:|--:|---|
| measured_superpath | 13 | 13125 | `float2int sroa loop-interchange gvn hotcoldsplit sroa argpromotion ipsccp dse sroa loop-interchange hotcoldsplit deadargelim` |
| strong_links | 12 | 13021 | `hotcoldsplit sroa attributor mem2reg gvn deadargelim early-cse ipsccp mldst-motion newgvn tailcallelim sink` |
| segment_tree | 16 | 13781 | `lower-expect mem2reg separate-const-offset-from-gep iroutliner jump-threading hotcoldsplit sroa correlated-propagation early-cse lower-expect mem2reg jump-threading correlated-propagation sroa loop-interchange gvn` |
| flow_paths | 6 | 11952 | `hotcoldsplit sroa loop-versioning loop-interchange instsimplify ipsccp` |
| cycle_breaking | 10 | 12597 | `loop-instsimplify hotcoldsplit sroa loop-interchange gvn loop-idiom loop-simplify mem2reg sccp constraint-elimination` |
| bucket_dag_teacher | 6 | 12496 | `sroa loop-interchange loop-versioning instsimplify ipsccp early-cse` |
| random_walk_1000 | 6 | 10984 | `jump-threading hotcoldsplit instcombine sroa loop-interchange instsimplify` |

## ispell

| эвристика | len | −байт | последовательность пассов |
|---|--:|--:|---|
| measured_superpath | 7 | 11042 | `constmerge sroa mem2reg newgvn mem2reg newgvn jump-threading` |
| strong_links | 10 | 12042 | `attributor constmerge newgvn speculative-execution mem2reg newgvn sroa jump-threading mldst-motion dse` |
| segment_tree | 16 | 11621 | `mem2reg early-cse globalopt attributor newgvn mem2reg newgvn mem2reg early-cse globalopt jump-threading constmerge sroa mem2reg early-cse gvn-sink` |
| flow_paths | 6 | 11100 | `float2int attributor jump-threading globaldce newgvn sroa` |
| cycle_breaking | 7 | 9898 | `loop-bound-split slp-vectorizer gvn-sink sroa partially-inline-libcalls constmerge gvn` |
| bucket_dag_teacher | 5 | 10767 | `newgvn jump-threading sroa globalopt attributor` |
| random_walk_1000 | 2 | 7532 | `mem2reg newgvn` |

## jpeg-c

| эвристика | len | −байт | последовательность пассов |
|---|--:|--:|---|
| measured_superpath | 20 | 41996 | `licm gvn sroa gvn-hoist gvn sroa newgvn speculative-execution loop-versioning-licm sroa newgvn gvn-sink libcalls-shrinkwrap gvn sroa float2int gvn-sink lcssa sroa jump-threading` |
| strong_links | 12 | 42843 | `constmerge newgvn gvn-sink simplifycfg sroa jump-threading newgvn mem2reg jump-threading gvn-sink early-cse gvn-hoist` |
| segment_tree | 16 | 42780 | `newgvn gvn-sink mem2reg simplifycfg newgvn gvn-sink sroa gvn-hoist lcssa sroa jump-threading newgvn lcssa jump-threading aggressive-instcombine gvn-sink` |
| flow_paths | 7 | 41476 | `jump-threading adce lower-expect mldst-motion loop-versioning-licm mem2reg newgvn` |
| cycle_breaking | 9 | 41628 | `simplifycfg sccp jump-threading mergereturn mldst-motion loop-versioning-licm mem2reg newgvn loop-interchange` |
| bucket_dag_teacher | 6 | 42020 | `sroa newgvn jump-threading loop-sink gvn-hoist early-cse` |
| random_walk_1000 | 11 | 41228 | `loop-simplifycfg newgvn lcssa indvars move-auto-init vector-combine loop-fusion slp-vectorizer reassociate mem2reg gvn-hoist` |

## jpeg-d

| эвристика | len | −байт | последовательность пассов |
|---|--:|--:|---|
| measured_superpath | 10 | 39980 | `gvn-sink mem2reg newgvn sroa simple-loop-unswitch sroa dfa-jump-threading attributor simple-loop-unswitch simplifycfg` |
| strong_links | 11 | 40894 | `mergefunc reassociate jump-threading dse simplifycfg newgvn sroa early-cse gvn-sink gvn-hoist mldst-motion` |
| segment_tree | 16 | 40492 | `simple-loop-unswitch sroa newgvn gvn-sink argpromotion newgvn sroa dfa-jump-threading newgvn sroa dfa-jump-threading speculative-execution gvn-hoist simplifycfg early-cse jump-threading` |
| flow_paths | 5 | 39748 | `mem2reg jump-threading mergereturn loop-idiom newgvn` |
| cycle_breaking | 9 | 39396 | `mergereturn aggressive-instcombine sroa nary-reassociate early-cse mem2reg newgvn dfa-jump-threading dse` |
| bucket_dag_teacher | 6 | 38316 | `sroa nary-reassociate early-cse dse alignment-from-assumptions jump-threading` |
| random_walk_1000 | 12 | 39755 | `indvars sroa dfa-jump-threading gvn-hoist infer-address-spaces slp-vectorizer constmerge bdce instcombine lcssa gvn-sink simplifycfg` |

## lame

| эвристика | len | −байт | последовательность пассов |
|---|--:|--:|---|
| measured_superpath | 8 | 38277 | `newgvn consthoist sroa early-cse jump-threading loop-versioning-licm sroa gvn-sink` |
| strong_links | 15 | 38359 | `mergefunc mem2reg sroa function-attrs newgvn simplifycfg loop-versioning-licm sroa constmerge sroa newgvn dse gvn-hoist mldst-motion deadargelim` |
| segment_tree | 14 | 38693 | `jump-threading sroa newgvn jump-threading loop-versioning-licm newgvn consthoist sroa gvn-sink jump-threading sroa newgvn globaldce gvn-hoist` |
| flow_paths | 7 | 37506 | `newgvn globaldce consthoist loop-instsimplify gvn-hoist mem2reg gvn-sink` |
| cycle_breaking | 12 | 37466 | `float2int lower-expect inferattrs partially-inline-libcalls attributor sroa function-attrs newgvn dfa-jump-threading consthoist loop-deletion dse` |
| bucket_dag_teacher | 4 | 37765 | `sroa early-cse jump-threading gvn-hoist` |
| random_walk_1000 | 11 | 38405 | `loop-instsimplify redundant-dbg-inst-elim sroa function-attrs newgvn jump-threading gvn-hoist lcssa mem2reg consthoist dse` |

## patricia

| эвристика | len | −байт | последовательность пассов |
|---|--:|--:|---|
| measured_superpath | 14 | 1694 | `simplifycfg div-rem-pairs globalopt gvn simplifycfg sroa globalopt gvn simplifycfg simple-loop-unswitch globalopt gvn simplifycfg simple-loop-unswitch` |
| strong_links | 6 | 1691 | `div-rem-pairs globalopt simplifycfg gvn simple-loop-unswitch sroa` |
| segment_tree | 12 | 1694 | `simplifycfg div-rem-pairs globalopt gvn simplifycfg sroa simplifycfg div-rem-pairs sroa gvn simplifycfg simple-loop-unswitch` |
| flow_paths | 3 | 1659 | `simplifycfg sroa gvn` |
| cycle_breaking | 6 | 1659 | `simplifycfg simple-loop-unswitch div-rem-pairs sroa globalopt gvn` |
| bucket_dag_teacher | 2 | 1216 | `simplifycfg gvn` |
| random_walk_1000 | 5 | 1659 | `simplifycfg simple-loop-unswitch div-rem-pairs sroa gvn` |

## qsort

| эвристика | len | −байт | последовательность пассов |
|---|--:|--:|---|
| measured_superpath | 3 | 228 | `sroa loop-load-elim loop-unroll-and-jam` |
| strong_links | 3 | 228 | `loop-load-elim sroa loop-unroll-and-jam` |
| segment_tree | 3 | 228 | `sroa loop-load-elim loop-unroll-and-jam` |
| flow_paths | 3 | 228 | `sroa loop-load-elim loop-unroll-and-jam` |
| cycle_breaking | 3 | 228 | `sroa loop-load-elim loop-unroll-and-jam` |
| bucket_dag_teacher | 1 | 132 | `sroa` |
| random_walk_1000 | 3 | 228 | `sroa loop-load-elim loop-unroll-and-jam` |

## rijndael

| эвристика | len | −байт | последовательность пассов |
|---|--:|--:|---|
| measured_superpath | 5 | 2297 | `loop-bound-split loop-load-elim mergeicmps sroa loop-bound-split` |
| strong_links | 1 | 2281 | `sroa` |
| segment_tree | 5 | 2297 | `loop-bound-split loop-load-elim loop-unroll sroa loop-bound-split` |
| flow_paths | 2 | 2126 | `instcombine sroa` |
| cycle_breaking | 4 | 2281 | `mergeicmps slp-vectorizer loop-unroll sroa` |
| bucket_dag_teacher | 3 | 2281 | `slp-vectorizer loop-unroll sroa` |
| random_walk_1000 | 6 | 2126 | `loop-bound-split loop-load-elim mergeicmps loop-unroll instcombine sroa` |

## sha

| эвристика | len | −байт | последовательность пассов |
|---|--:|--:|---|
| measured_superpath | 4 | 506 | `aggressive-instcombine chr mem2reg loop-interchange` |
| strong_links | 1 | 449 | `mem2reg` |
| segment_tree | 16 | 538 | `aggressive-instcombine chr mem2reg lcssa mem2reg loop-interchange move-auto-init sccp mem2reg lcssa move-auto-init sccp mem2reg move-auto-init loop-vectorize sccp` |
| flow_paths | 2 | 506 | `mem2reg loop-interchange` |
| cycle_breaking | 7 | 538 | `mem2reg loop-interchange div-rem-pairs move-auto-init loop-vectorize lcssa sccp` |
| bucket_dag_teacher | 3 | 538 | `mem2reg loop-vectorize sccp` |
| random_walk_1000 | 6 | 538 | `mem2reg loop-interchange lcssa move-auto-init loop-vectorize sccp` |

## stringsearch

| эвристика | len | −байт | последовательность пассов |
|---|--:|--:|---|
| measured_superpath | 9 | 1035 | `bdce simplifycfg newgvn mem2reg gvn-hoist newgvn mem2reg lower-constant-intrinsics loop-unroll-and-jam` |
| strong_links | 4 | 971 | `attributor newgvn simplifycfg mem2reg` |
| segment_tree | 7 | 1035 | `simplifycfg newgvn mem2reg gvn-hoist newgvn mem2reg loop-simplifycfg` |
| flow_paths | 4 | 1019 | `simplifycfg newgvn mem2reg loop-simplifycfg` |
| cycle_breaking | 8 | 1019 | `bdce loop-unroll-and-jam mldst-motion simplifycfg newgvn mem2reg lower-constant-intrinsics loop-simplifycfg` |
| bucket_dag_teacher | 4 | 955 | `simplifycfg newgvn aggressive-instcombine sroa` |
| random_walk_1000 | 5 | 845 | `loop-unroll-and-jam mldst-motion simplifycfg aggressive-instcombine sroa` |

## stringsearch2

| эвристика | len | −байт | последовательность пассов |
|---|--:|--:|---|
| measured_superpath | 4 | 443 | `tailcallelim loop-load-elim mem2reg jump-threading` |
| strong_links | 2 | 443 | `jump-threading mem2reg` |
| segment_tree | 4 | 443 | `argpromotion loop-load-elim mem2reg jump-threading` |
| flow_paths | 2 | 443 | `mem2reg jump-threading` |
| cycle_breaking | 8 | 434 | `instcombine lower-constant-intrinsics function-attrs argpromotion tailcallelim loop-load-elim mem2reg jump-threading` |
| bucket_dag_teacher | 2 | 379 | `lower-constant-intrinsics mem2reg` |
| random_walk_1000 | 6 | 434 | `instcombine lower-constant-intrinsics argpromotion tailcallelim mem2reg jump-threading` |

## susan

| эвристика | len | −байт | последовательность пассов |
|---|--:|--:|---|
| measured_superpath | 8 | 12525 | `loop-idiom newgvn attributor ipsccp consthoist bdce sroa iroutliner` |
| strong_links | 6 | 12846 | `newgvn mem2reg iroutliner attributor sroa correlated-propagation` |
| segment_tree | 16 | 13725 | `bdce sroa tailcallelim separate-const-offset-from-gep loop-load-elim adce sroa iroutliner indvars globaldce sroa iroutliner loop-load-elim globaldce sroa iroutliner` |
| flow_paths | 4 | 11890 | `mergereturn globaldce sroa iroutliner` |
| cycle_breaking | 7 | 10119 | `consthoist bdce sroa tailcallelim constmerge aggressive-instcombine separate-const-offset-from-gep` |
| bucket_dag_teacher | 3 | 11885 | `loop-simplifycfg sroa iroutliner` |
| random_walk_1000 | 9 | 0 | `loop-idiom partially-inline-libcalls attributor ipsccp redundant-dbg-inst-elim float2int loop-sink aggressive-instcombine separate-const-offset-from-gep` |

## tiff2bw

| эвристика | len | −байт | последовательность пассов |
|---|--:|--:|---|
| measured_superpath | 15 | 38998 | `jump-threading sroa sink licm nary-reassociate newgvn sroa jump-threading libcalls-shrinkwrap iroutliner mem2reg ipsccp mergereturn sroa licm` |
| strong_links | 10 | 38172 | `attributor jump-threading mem2reg sroa newgvn ipsccp gvn-sink gvn-hoist instsimplify indvars` |
| segment_tree | 16 | 39419 | `correlated-propagation jump-threading sroa newgvn mem2reg loop-bound-split ipsccp jump-threading newgvn sroa jump-threading iroutliner gvn-sink sroa licm jump-threading` |
| flow_paths | 6 | 36977 | `newgvn consthoist adce sroa licm jump-threading` |
| cycle_breaking | 12 | 37211 | `separate-const-offset-from-gep ipsccp speculative-execution sroa newgvn sink loop-distribute lower-constant-intrinsics jump-threading gvn nary-reassociate iroutliner` |
| bucket_dag_teacher | 9 | 36978 | `sroa ipsccp newgvn div-rem-pairs loop-versioning-licm loop-sink iroutliner mergereturn constmerge` |
| random_walk_1000 | 11 | 38752 | `div-rem-pairs early-cse gvn-hoist simplifycfg jump-threading gvn-sink sroa tailcallelim gvn instcombine iroutliner` |

## tiff2rgba

| эвристика | len | −байт | последовательность пассов |
|---|--:|--:|---|
| measured_superpath | 19 | 38859 | `nary-reassociate jump-threading sroa simple-loop-unswitch consthoist partially-inline-libcalls sroa jump-threading sroa gvn iroutliner jump-threading sroa loop-flatten sink globaldce mergefunc mem2reg gvn-hoist` |
| strong_links | 16 | 39688 | `mem2reg gvn-hoist jump-threading memcpyopt ipsccp correlated-propagation jump-threading dse mem2reg attributor gvn gvn-sink gvn-hoist iroutliner aggressive-instcombine indvars` |
| segment_tree | 16 | 39090 | `jump-threading sroa gvn iroutliner jump-threading sroa newgvn iroutliner jump-threading sroa newgvn gvn-sink newgvn jump-threading sroa licm` |
| flow_paths | 7 | 36193 | `loop-flatten attributor speculative-execution newgvn sroa jump-threading aggressive-instcombine` |
| cycle_breaking | 12 | 37049 | `vector-combine strip-dead-prototypes jump-threading sroa loop-flatten simplifycfg attributor dfa-jump-threading dse adce gvn iroutliner` |
| bucket_dag_teacher | 9 | 37350 | `mem2reg gvn aggressive-instcombine simplifycfg strip-dead-prototypes reassociate loop-flatten sink iroutliner` |
| random_walk_1000 | 12 | 36754 | `sccp mem2reg ipsccp aggressive-instcombine reassociate newgvn sroa gvn-hoist gvn simplifycfg attributor iroutliner` |

## tiffdither

| эвристика | len | −байт | последовательность пассов |
|---|--:|--:|---|
| measured_superpath | 17 | 38503 | `gvn-hoist sroa jump-threading loop-flatten loop-versioning-licm newgvn mem2reg simplifycfg loop-interchange sroa mergefunc mergereturn dse mem2reg simplifycfg sroa iroutliner` |
| strong_links | 16 | 39624 | `correlated-propagation gvn-hoist mergefunc memcpyopt jump-threading sroa gvn simplifycfg ipsccp early-cse gvn-sink attributor indvars iroutliner loop-instsimplify constmerge` |
| segment_tree | 16 | 38725 | `early-cse simplifycfg sroa jump-threading gvn-sink loop-interchange jump-threading newgvn mem2reg simplifycfg gvn-hoist loop-interchange sroa jump-threading constmerge aggressive-instcombine` |
| flow_paths | 5 | 37025 | `sroa argpromotion newgvn jump-threading loop-flatten` |
| cycle_breaking | 10 | 37682 | `newgvn memcpyopt mem2reg jump-threading consthoist sccp ipsccp sink early-cse iroutliner` |
| bucket_dag_teacher | 10 | 37230 | `sroa slp-vectorizer ipsccp early-cse loop-bound-split loop-simplify simplifycfg elim-avail-extern attributor iroutliner` |
| random_walk_1000 | 10 | 33889 | `sroa dce loop-simplify licm elim-avail-extern separate-const-offset-from-gep simple-loop-unswitch bdce loop-bound-split iroutliner` |

## tiffmedian

| эвристика | len | −байт | последовательность пассов |
|---|--:|--:|---|
| measured_superpath | 20 | 41226 | `gvn-sink float2int sroa newgvn dce called-value-propagation loop-deletion sroa jump-threading gvn-sink sroa alignment-from-assumptions elim-avail-extern loop-predication float2int sroa gvn-sink float2int sroa iroutliner` |
| strong_links | 16 | 41044 | `simplifycfg sroa loop-predication gvn instcombine loop-idiom sroa gvn-sink sroa deadargelim sccp slp-vectorizer float2int sroa ipsccp iroutliner` |
| segment_tree | 14 | 42505 | `float2int sroa slp-vectorizer jump-threading vector-combine sroa newgvn gvn-hoist instcombine gvn-sink sroa iroutliner loop-flatten mergereturn` |
| flow_paths | 4 | 39553 | `jump-threading sroa newgvn licm` |
| cycle_breaking | 11 | 39150 | `gvn-sink sroa simplifycfg ipsccp mem2reg sink gvn instcombine loop-rotate attributor iroutliner` |
| bucket_dag_teacher | 8 | 38004 | `mem2reg aggressive-instcombine gvn-hoist speculative-execution simplifycfg attributor constmerge loop-bound-split` |
| random_walk_1000 | 3 | 34498 | `float2int mergeicmps sroa` |

