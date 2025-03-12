library(tidyverse)
library(openxlsx)
library(arrow)
library(ggExtra)

setwd("C:/Users/idwe/Documents/Github/Analyzing-concussions-through-eyetracking-measurements")

metadata <- read.xlsx("data/demographic_info.xlsx") %>% 
  mutate(participant_id = as.character(ID)) %>% 
  mutate(eye_tracking_date = as.Date(Eye.tracking.date, origin = "1899-12-30")) %>% 
  select(participant_id, group=Group, age=Age, gender=Gender, eye_tracking_date)

# Anti saccade ----

anti_saccade <- read_parquet("data/processed/ANTI_SACCADE.pq")

## Left and right eyes ----
left_eye_anti_saccade_filtered <- anti_saccade %>% 
  filter(!(event %in% c("SSACC", "SFIX"))) %>% 
  filter(eye %in% c(NA, "L")) %>% 
  mutate(eye = "L") %>% 
  mutate(time = coalesce(time, end_time)) %>% 
  arrange(participant_id, trial_id, time, eye) %>% 
  group_by(participant_id, trial_id) %>% 
  mutate(time = time - min(time)) %>% 
  mutate(side = first(na.omit(side))) %>% 
  fill(colour, .direction = "down") %>% 
  fill(stimulus_x, .direction = "down") %>% 
  fill(stimulus_y, .direction = "down") %>% 
  ungroup()


right_eye_anti_saccade_filtered <- anti_saccade %>% 
  filter(!(event %in% c("SSACC", "SFIX"))) %>% 
  filter(eye %in% c(NA, "R")) %>% 
  mutate(eye = "R") %>% 
  mutate(time = coalesce(time, end_time)) %>% 
  arrange(participant_id, trial_id, time, eye) %>% 
  group_by(participant_id, trial_id) %>% 
  mutate(time = time - min(time)) %>% 
  mutate(side = first(na.omit(side))) %>% 
  fill(colour, .direction = "down") %>% 
  fill(stimulus_x, .direction = "down") %>% 
  fill(stimulus_y, .direction = "down") %>% 
  ungroup()


## is the trial correct? ----

left_eye_anti_saccade_filtered %>% 
  filter(event %in% c("FIXPOINT", "EFIX", "START", "END", "TRIALID", "TRIAL_VAR_DATA")) %>% 
  mutate(gaze_direction = case_when(
    (colour == "255 0 0" & event == "EFIX" & x > 1100) ~ "right",
    (colour == "255 0 0" & event == "EFIX" & x < 820) ~ "left",
    (colour == "255 0 0" & event == "EFIX") ~ "middle",
    T ~ NA
  )) %>% 
  mutate(correct_gaze_direction = case_when(
    side == gaze_direction ~ F,
    T ~ T
  )) %>% 
  group_by(participant_id, trial_id) %>% 
  summarise(is_trial_incorrect = if_else(sum(!correct_gaze_direction) >= 1, T, F)
            ) %>% 
  ungroup() %>% 
  left_join(metadata) %>% 
  count(group,is_trial_incorrect) %>% 
  View()


right_eye_anti_saccade_filtered %>% 
  filter(event %in% c("FIXPOINT", "EFIX", "START", "END", "TRIALID", "TRIAL_VAR_DATA")) %>% 
  mutate(gaze_direction = case_when(
    (colour == "255 0 0" & event == "EFIX" & x > 1100) ~ "right",
    (colour == "255 0 0" & event == "EFIX" & x < 820) ~ "left",
    (colour == "255 0 0" & event == "EFIX") ~ "middle",
    T ~ NA
  )) %>% 
  mutate(correct_gaze_direction = case_when(
    side == gaze_direction ~ F,
    T ~ T
  )) %>% 
  group_by(participant_id, trial_id) %>% 
  summarise(is_trial_incorrect = if_else(sum(!correct_gaze_direction) >= 1, T, F)
  ) %>% 
  ungroup() %>% 
  left_join(metadata) %>% 
  count(group,is_trial_incorrect) %>% 
  View()


# Fitt's law ----


fitts_law <- read_parquet("data/processed/FITTS_LAW.pq")




## Left and right eyes ----
left_eye_fitts_law_filtered <- fitts_law %>% 
  filter(!(event %in% c("SSACC", "SFIX"))) %>% 
  filter(eye %in% c(NA, "L")) %>% 
  mutate(eye = "L") %>% 
  mutate(time = coalesce(time, end_time)) %>% 
  arrange(participant_id, trial_id, time, eye) %>% 
  group_by(participant_id, trial_id) %>% 
  mutate(time = time - min(time)) %>% 
  ungroup()


right_eye_fitts_law_filtered <- fitts_law %>% 
  filter(!(event %in% c("SSACC", "SFIX"))) %>% 
  filter(eye %in% c(NA, "R")) %>% 
  mutate(eye = "R") %>% 
  mutate(time = coalesce(time, end_time)) %>% 
  arrange(participant_id, trial_id, time, eye) %>% 
  group_by(participant_id, trial_id) %>% 
  mutate(time = time - min(time)) %>% 
  ungroup()


right_eye_fitts_law_filtered %>% 
  group_by(participant_id, trial_id) %>% 
  filter(event == "EFIX") %>% 
  summarise(n_fixations = n()) %>% 
  group_by(participant_id) %>% 
  summarise(avg_fix_pr_trial = mean(n_fixations)) %>% 
  left_join(metadata) %>% 
  group_by(group) %>% 
  summarise(mean(avg_fix_pr_trial)) %>% 
  View()
  


# reaction ----

reaction <- read_parquet("data/processed/REACTION.pq")


left_eye_reaction_filtered <- reaction %>% 
  filter(!(event %in% c("SSACC", "SFIX"))) %>% 
  filter(eye %in% c(NA, "L")) %>% 
  mutate(eye = "L") %>% 
  mutate(time = coalesce(time, end_time)) %>% 
  arrange(participant_id, trial_id, time, eye) %>% 
  group_by(participant_id, trial_id) %>% 
  mutate(time = time - min(time)) %>% 
  ungroup()



left_eye_reaction_filtered %>% 
  filter(event =="FIXPOINT") %>% 
  count(participant_id) %>% 
  View()


# fixations ----


fixations <- read_parquet("data/processed/FIXATIONS.pq")



left_eye_fixations_filtered <- fixations %>% 
  filter(!(event %in% c("SSACC", "SFIX"))) %>% 
  filter(eye %in% c(NA, "L")) %>% 
  mutate(eye = "L") %>% 
  mutate(time = coalesce(time, end_time)) %>% 
  arrange(participant_id, trial_id, time, eye) %>% 
  group_by(participant_id, trial_id) %>% 
  mutate(time = time - min(time)) %>% 
  ungroup()


# king devick ----

king_devick <- read_parquet("data/processed/KING_DEVICK.pq")


left_eye_king_devick_filtered <- king_devick %>% 
  filter(!(event %in% c("SSACC", "SFIX"))) %>% 
  filter(eye %in% c(NA, "L")) %>% 
  mutate(eye = "L") %>% 
  mutate(time = coalesce(time, end_time)) %>% 
  arrange(participant_id, trial_id, time, eye) %>% 
  group_by(participant_id, trial_id) %>% 
  mutate(time = time - min(time)) %>% 
  ungroup()


left_eye_king_devick_filtered %>% 
  count(marks) %>% 
  View()


# shapes ----

shapes <- read_parquet("data/processed/SHAPES.pq")


left_eye_shapes_filtered <- shapes %>% 
  filter(!(event %in% c("SSACC", "SFIX"))) %>% 
  filter(eye %in% c(NA, "L")) %>% 
  mutate(eye = "L") %>% 
  mutate(time = coalesce(time, end_time)) %>% 
  arrange(participant_id, trial_id, time, eye) %>% 
  group_by(participant_id, trial_id) %>% 
  mutate(time = time - min(time)) %>% 
  ungroup()

left_eye_shapes_filtered %>% 
  filter(event == "FIXPOINT") %>% 
  nrow()
# evil bastard ----

evil_bastard <- read_parquet("data/processed/EVIL_BASTARD.pq")


left_eye_evil_bastard_filtered <- evil_bastard %>%
  filter(!(event %in% c("SSACC", "SFIX"))) %>% 
  filter(eye %in% c(NA, "L")) %>% 
  mutate(eye = "L") %>% 
  mutate(time = coalesce(time, end_time)) %>% 
  arrange(participant_id, trial_id, time, eye) %>% 
  group_by(participant_id, trial_id) %>% 
  mutate(time = time - min(time)) %>% 
  ungroup()



left_eye_evil_bastard_filtered %>% 
  filter(event == "FIXPOINT") %>% 
  nrow()


# smooth pursuits ----

smooth_pursuits <- read_parquet("data/processed/SMOOTH_PURSUITS.pq")


left_eye_smooth_pursuits_filtered <- smooth_pursuits %>%
  filter(!(event %in% c("SSACC", "SFIX"))) %>% 
  filter(eye %in% c(NA, "L")) %>% 
  mutate(eye = "L") %>% 
  mutate(time = coalesce(time, end_time)) %>% 
  arrange(participant_id, trial_id, time, eye) %>% 
  group_by(participant_id, trial_id) %>% 
  mutate(time = time - min(time)) %>% 
  ungroup()



left_eye_smooth_pursuits_filtered %>% 
  filter(event == "FIXPOINT") %>% 
  nrow()




