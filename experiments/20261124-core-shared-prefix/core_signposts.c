#include <os/log.h>
#include <os/signpost.h>

static os_log_t core_log;

void core_signposts_init(void) {
    core_log = os_log_create("org.gpt_lab.core", OS_LOG_CATEGORY_POINTS_OF_INTEREST);
}

uint64_t core_signpost_begin(int phase) {
    os_signpost_id_t id = os_signpost_id_generate(core_log);
    switch (phase) {
    case 0: os_signpost_interval_begin(core_log, id, "CORE sweep"); break;
    case 1: os_signpost_interval_begin(core_log, id, "CPU collation"); break;
    case 2: os_signpost_interval_begin(core_log, id, "Model submission"); break;
    }
    return id;
}

void core_signpost_end(int phase, uint64_t id) {
    switch (phase) {
    case 0: os_signpost_interval_end(core_log, id, "CORE sweep"); break;
    case 1: os_signpost_interval_end(core_log, id, "CPU collation"); break;
    case 2: os_signpost_interval_end(core_log, id, "Model submission"); break;
    }
}
