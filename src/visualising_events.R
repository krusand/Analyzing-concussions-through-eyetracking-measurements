library(dplyr)
library(openxlsx)
library(arrow)
library(ggplot2)
library(tidyr)
library(ggExtra)
library(plotly)
options(scipen=999)

setwd("/Users/viscom2025/Documents/Github/Analyzing-concussions-through-eyetracking-measurements")


# Anti saccade ----

anti_saccade_raw <- read_parquet("data/processed/ANTI_SACCADE.pq") %>% 
  group_by(participant_id, trial_id) %>% 
  mutate(fixpoint_white_time = case_when(
    event == 'FIXPOINT' & colour == '255 255 255' ~ time
  )) %>% 
  mutate(fixpoint_white_time = first(na.omit(fixpoint_white_time))) %>% 
  mutate(time_elapsed = coalesce(first(na.omit(delay)), first(na.omit(time_elapsed)))) %>% 
  mutate(time = case_when(
    event == 'FIXPOINT' & colour == '255 0 0' ~ as.numeric(fixpoint_white_time) + 1000*as.numeric(time_elapsed),
    T ~ time
  )) %>% 
  ungroup()


## Cleaning ----

anti_saccade_right <- anti_saccade_raw %>% 
  filter(!(event %in% c("SSACC", "SFIX"))) %>% 
  filter(eye %in% c(NA, "R")) %>% 
  mutate(m_eye = "R") %>% 
  mutate(m_time = coalesce(time, end_time)) %>% 
  arrange(participant_id, trial_id, m_time) %>% 
  group_by(participant_id, trial_id) %>% 
  fill(c(colour, stimulus_x, stimulus_y), .direction = "down") %>% 
  mutate(m_stand_time = m_time - min(m_time)) %>% 
  mutate(m_stand_start_time = start_time - min(m_time)) %>% 
  mutate(m_stand_end_time = end_time - min(m_time)) %>% 
  mutate(m_stimulus_side = first(na.omit(side))) %>% 
  mutate(m_stimulus_delay = coalesce( first(na.omit(time_elapsed)) , first(na.omit(delay)) )) %>% 
  mutate(m_stimulus_active = case_when(
    colour == "255 255 255" ~ F,
    colour == "255 0 0" ~ T,
    T ~ NA
  )) %>% 
  filter(!is.na(m_stimulus_active)) %>%
  ungroup() %>% 
  select(experiment
         , participant_id
         , trial_id
         , stand_time=m_stand_time
         , eye = m_eye
         , event
         , delay = m_stimulus_delay
         , stimulus_active = m_stimulus_active
         , stimulus_colour = colour
         , stimulus_x
         , stimulus_y
         , fix_x = x
         , fix_y = y
         , sacc_start_x = start_x
         , sacc_start_y = start_y
         , sacc_end_x = end_x
         , sacc_end_y = end_y
         , stand_start_time = m_stand_start_time 
         , stand_end_time = m_stand_end_time
         , avg_pupil_size
         , peak_velocity
         , amplitude
         , duration)


anti_saccade_left <- anti_saccade_raw %>% 
  filter(!(event %in% c("SSACC", "SFIX"))) %>% 
  filter(eye %in% c(NA, "L")) %>% 
  mutate(m_eye = "L") %>% 
  mutate(m_time = coalesce(time, end_time)) %>% 
  arrange(participant_id, trial_id, m_time) %>% 
  group_by(participant_id, trial_id) %>% 
  fill(c(colour, stimulus_x, stimulus_y), .direction = "down") %>% 
  mutate(m_stand_time = m_time - min(m_time)) %>% 
  mutate(m_stand_start_time = start_time - min(m_time)) %>% 
  mutate(m_stand_end_time = end_time - min(m_time)) %>% 
  mutate(m_stimulus_side = first(na.omit(side))) %>% 
  mutate(m_stimulus_delay = coalesce( first(na.omit(time_elapsed)) , first(na.omit(delay)) )) %>% 
  mutate(m_stimulus_active = case_when(
    colour == "255 255 255" ~ F,
    colour == "255 0 0" ~ T,
    T ~ NA
  )) %>% 
  filter(!is.na(m_stimulus_active)) %>%
  ungroup() %>% 
  select(experiment
         , participant_id
         , trial_id
         , stand_time=m_stand_time
         , eye = m_eye
         , event
         , delay = m_stimulus_delay
         , stimulus_active = m_stimulus_active
         , stimulus_colour = colour
         , stimulus_x
         , stimulus_y
         , fix_x = x
         , fix_y = y
         , sacc_start_x = start_x
         , sacc_start_y = start_y
         , sacc_end_x = end_x
         , sacc_end_y = end_y
         , stand_start_time = m_stand_start_time 
         , stand_end_time = m_stand_end_time
         , avg_pupil_size
         , peak_velocity
         , amplitude
         , duration)


anti_saccade <- anti_saccade_left %>% 
  bind_rows(anti_saccade_right)



## Visualising ----

p_id <- 395
t_id <- 0

anti_saccade_plotting_data <- anti_saccade %>% 
  filter(participant_id == p_id , trial_id == t_id) %>% 
  filter(event %in% c("FIXPOINT", "EFIX") ) %>% 
  mutate(x_plotting = coalesce(fix_x, stimulus_x),
         y_plotting = coalesce(fix_y, stimulus_y),
         colour_plotting = case_when(
           stimulus_colour == "255 0 0" & event == "FIXPOINT" ~ "Stimulus",
           stimulus_colour == "255 255 255" & event == "FIXPOINT" ~ "Middle stimulus",
           stimulus_colour == "255 0 0" & event == "EFIX" ~ "Fixations - Post stimulus",
           stimulus_colour == "255 255 255" & event == "EFIX" ~ "Fixations - Pre stimulus",
         )) %>% 
  group_by(eye,colour_plotting) %>% 
  mutate(event_nr = row_number()) %>% 
  ungroup() %>% 
  filter(!is.na(stimulus_active))



p <- ggplot(data = anti_saccade_plotting_data, 
       aes(x=x_plotting, y=y_plotting, colour = colour_plotting, 
           label = event_nr)) + 
  geom_point() +
  geom_path() +
  facet_wrap(~eye)

ggplotly(p, tooltip = c("label", "colour_plotting"))


# Reactions ----
# Special participants are only relevant for reactions experiment. These denote participants where the data is not complete (there are no fixpoints)
special_participants <- c(87, 89, 93, 96, 103, 105, 109, 117, 118, 119, 120, 127, 128, 141)


reactions_raw <- read_parquet("data/processed/REACTION.pq") %>% 
  filter(!(participant_id %in% special_participants)) %>% 
  mutate(stimulus_x = coalesce(stimulus_x, as.numeric(pos_x)),
         stimulus_y = coalesce(stimulus_y, as.numeric(pos_y))) %>% 
  select(-c(pos_x,pos_y))



## Cleaning ----

reactions_right <- reactions_raw %>% 
  filter(!(event %in% c("SSACC", "SFIX"))) %>% 
  filter(eye %in% c(NA, "R")) %>% 
  mutate(m_eye = "R") %>% 
  mutate(m_time = coalesce(time, end_time)) %>% 
  arrange(participant_id, trial_id, m_time) %>% 
  group_by(participant_id, trial_id) %>% 
  fill(c(colour, stimulus_x, stimulus_y), .direction = "down") %>% 
  mutate(m_stand_time = m_time - min(m_time)) %>% 
  mutate(m_stand_start_time = start_time - min(m_time)) %>% 
  mutate(m_stand_end_time = end_time - min(m_time)) %>% 
  mutate(m_stimulus_delay = first(na.omit(delay))) %>% 
  mutate(m_stimulus_active = case_when(
    colour == "255 255 255" ~ F,
    colour == "255 0 0" ~ T,
    T ~ NA
  )) %>% 
  filter(!is.na(m_stimulus_active)) %>%
  ungroup() %>% 
  select(experiment
         , participant_id
         , trial_id
         , stand_time=m_stand_time
         , eye = m_eye
         , event
         , delay = m_stimulus_delay
         , stimulus_active = m_stimulus_active
         , stimulus_colour = colour
         , stimulus_x
         , stimulus_y
         , fix_x = x
         , fix_y = y
         , sacc_start_x = start_x
         , sacc_start_y = start_y
         , sacc_end_x = end_x
         , sacc_end_y = end_y
         , stand_start_time = m_stand_start_time 
         , stand_end_time = m_stand_end_time
         , avg_pupil_size
         , peak_velocity
         , amplitude
         , duration)

reactions_left <- reactions_raw %>% 
  filter(!(event %in% c("SSACC", "SFIX"))) %>% 
  filter(eye %in% c(NA, "L")) %>% 
  mutate(m_eye = "L") %>% 
  mutate(m_time = coalesce(time, end_time)) %>% 
  arrange(participant_id, trial_id, m_time) %>% 
  group_by(participant_id, trial_id) %>% 
  fill(c(colour, stimulus_x, stimulus_y), .direction = "down") %>% 
  mutate(m_stand_time = m_time - min(m_time)) %>% 
  mutate(m_stand_start_time = start_time - min(m_time)) %>% 
  mutate(m_stand_end_time = end_time - min(m_time)) %>% 
  mutate(m_stimulus_delay = first(na.omit(delay))) %>% 
  mutate(m_stimulus_active = case_when(
    colour == "255 255 255" ~ F,
    colour == "255 0 0" ~ T,
    T ~ NA
  )) %>% 
  filter(!is.na(m_stimulus_active)) %>%
  ungroup() %>% 
  select(experiment
         , participant_id
         , trial_id
         , stand_time=m_stand_time
         , eye = m_eye
         , event
         , delay = m_stimulus_delay
         , stimulus_active = m_stimulus_active
         , stimulus_colour = colour
         , stimulus_x
         , stimulus_y
         , fix_x = x
         , fix_y = y
         , sacc_start_x = start_x
         , sacc_start_y = start_y
         , sacc_end_x = end_x
         , sacc_end_y = end_y
         , stand_start_time = m_stand_start_time 
         , stand_end_time = m_stand_end_time
         , avg_pupil_size
         , peak_velocity
         , amplitude
         , duration)


reactions <- reactions_left %>% 
  bind_rows(reactions_right)


## Visualising ----


p_id <- 237
t_id <- 1

reactions_plotting_data <- reactions %>% 
  filter(participant_id == p_id , trial_id == t_id) %>% 
  filter(event %in% c("FIXPOINT", "EFIX") ) %>% 
  mutate(x_plotting = coalesce(fix_x, stimulus_x),
         y_plotting = coalesce(fix_y, stimulus_y),
         colour_plotting = case_when(
           stimulus_colour == "255 0 0" & event == "FIXPOINT" ~ "Stimulus",
           stimulus_colour == "255 255 255" & event == "FIXPOINT" ~ "Middle stimulus",
           stimulus_colour == "255 0 0" & event == "EFIX" ~ "Fixations - Post stimulus",
           stimulus_colour == "255 255 255" & event == "EFIX" ~ "Fixations - Pre stimulus",
         )) %>% 
  group_by(eye,colour_plotting) %>% 
  mutate(event_nr = row_number()) %>% 
  ungroup() %>% 
  filter(!is.na(stimulus_active))



p <- ggplot(data = reactions_plotting_data, 
            aes(x=x_plotting, y=y_plotting, colour = colour_plotting, 
                label = event_nr)) + 
  geom_point() +
  geom_path() +
  facet_wrap(~eye)

ggplotly(p, tooltip = c("label", "colour_plotting"))

# King Devick ----

king_devick_raw <- read_parquet("data/processed/KING_DEVICK.pq")


## Cleaning ----

king_devick_right <- king_devick_raw %>% 
  filter(!(event %in% c("SSACC", "SFIX"))) %>% 
  filter(eye %in% c(NA, "R")) %>% 
  mutate(m_eye = "R") %>% 
  mutate(m_time = coalesce(time, end_time)) %>% 
  arrange(participant_id, trial_id, m_time) %>% 
  group_by(participant_id, trial_id) %>% 
  fill(time_elapsed, .direction = "downup") %>% 
  mutate(m_stand_time = m_time - min(m_time)) %>% 
  mutate(m_stand_start_time = start_time - min(m_time)) %>% 
  mutate(m_stand_end_time = end_time - min(m_time)) %>% 
  ungroup() %>% 
  select(experiment
         , participant_id
         , trial_id
         , time_elapsed
         , stand_time=m_stand_time
         , eye = m_eye
         , event
         , fix_x = x
         , fix_y = y
         , sacc_start_x = start_x
         , sacc_start_y = start_y
         , sacc_end_x = end_x
         , sacc_end_y = end_y
         , stand_start_time = m_stand_start_time 
         , stand_end_time = m_stand_end_time
         , avg_pupil_size
         , peak_velocity
         , amplitude
         , duration)

king_devick_left <- king_devick_raw %>% 
  filter(!(event %in% c("SSACC", "SFIX"))) %>% 
  filter(eye %in% c(NA, "L")) %>% 
  mutate(m_eye = "L") %>% 
  mutate(m_time = coalesce(time, end_time)) %>% 
  arrange(participant_id, trial_id, m_time) %>% 
  group_by(participant_id, trial_id) %>% 
  fill(time_elapsed, .direction = "downup") %>% 
  mutate(m_stand_time = m_time - min(m_time)) %>% 
  mutate(m_stand_start_time = start_time - min(m_time)) %>% 
  mutate(m_stand_end_time = end_time - min(m_time)) %>% 
  ungroup() %>% 
  select(experiment
         , participant_id
         , trial_id
         , time_elapsed
         , stand_time=m_stand_time
         , eye = m_eye
         , event
         , fix_x = x
         , fix_y = y
         , sacc_start_x = start_x
         , sacc_start_y = start_y
         , sacc_end_x = end_x
         , sacc_end_y = end_y
         , stand_start_time = m_stand_start_time 
         , stand_end_time = m_stand_end_time
         , avg_pupil_size
         , peak_velocity
         , amplitude
         , duration)


king_devick <- king_devick_left %>% 
  bind_rows(king_devick_right)

## Visualising ----


p_id <- 237
t_id <- 2

king_devick_plotting_data <- king_devick %>% 
  filter(participant_id == p_id , trial_id == t_id) %>% 
  mutate(time_elapsed_ms = as.numeric(time_elapsed) * 1000) %>% 
  group_by(event) %>% 
  filter(event %in% c("EFIX") & stand_time < time_elapsed_ms ) %>% 
  mutate(x_plotting = fix_x,
         y_plotting = fix_y) %>% 
  group_by(eye) %>% 
  mutate(event_nr = row_number()) %>% 
  ungroup() 



p <- ggplot(data = king_devick_plotting_data, 
            aes(x=x_plotting, y=y_plotting, label = event_nr)) + 
  geom_point() +
  geom_path() +
  facet_wrap(~eye)

ggplotly(p, tooltip = c("label"))



# Fitts Law ----



fitts_law_raw <- read_parquet("data/processed/FITTS_LAW.pq")

## Cleaning ----

fitts_law_right <- fitts_law_raw %>% 
  filter(!(event %in% c("SSACC", "SFIX"))) %>% 
  filter(eye %in% c(NA, "R")) %>% 
  mutate(m_eye = "R") %>% 
  mutate(m_time = coalesce(time, end_time)) %>% 
  arrange(participant_id, trial_id, m_time) %>% 
  group_by(participant_id, trial_id) %>% 
  fill(c(distance, target_width), .direction = "downup") %>% 
  mutate(m_stand_time = m_time - min(m_time)) %>% 
  mutate(m_stand_start_time = start_time - min(m_time)) %>% 
  mutate(m_stand_end_time = end_time - min(m_time)) %>% 
  ungroup() %>% 
  select(experiment
         , participant_id
         , trial_id
         , stand_time=m_stand_time
         , eye = m_eye
         , event
         , stimulus_colour = colour
         , stimulus_x
         , stimulus_y
         , fix_x = x
         , fix_y = y
         , sacc_start_x = start_x
         , sacc_start_y = start_y
         , sacc_end_x = end_x
         , sacc_end_y = end_y
         , stand_start_time = m_stand_start_time 
         , stand_end_time = m_stand_end_time
         , avg_pupil_size
         , peak_velocity
         , amplitude
         , duration)



fitts_law_left <- fitts_law_raw %>% 
  filter(!(event %in% c("SSACC", "SFIX"))) %>% 
  filter(eye %in% c(NA, "L")) %>% 
  mutate(m_eye = "L") %>% 
  mutate(m_time = coalesce(time, end_time)) %>% 
  arrange(participant_id, trial_id, m_time) %>% 
  group_by(participant_id, trial_id) %>% 
  fill(c(distance, target_width), .direction = "downup") %>% 
  mutate(m_stand_time = m_time - min(m_time)) %>% 
  mutate(m_stand_start_time = start_time - min(m_time)) %>% 
  mutate(m_stand_end_time = end_time - min(m_time)) %>% 
  ungroup() %>% 
  select(experiment
         , participant_id
         , trial_id
         , stand_time=m_stand_time
         , eye = m_eye
         , event
         , stimulus_colour = colour
         , stimulus_x
         , stimulus_y
         , fix_x = x
         , fix_y = y
         , sacc_start_x = start_x
         , sacc_start_y = start_y
         , sacc_end_x = end_x
         , sacc_end_y = end_y
         , stand_start_time = m_stand_start_time 
         , stand_end_time = m_stand_end_time
         , avg_pupil_size
         , peak_velocity
         , amplitude
         , duration)

fitts_law <- fitts_law_left %>% 
  bind_rows(fitts_law_right)





## Visualising ----

p_id <- 396
t_id <- 4

fitts_law_plotting_data <- fitts_law %>% 
  filter(participant_id == p_id , trial_id == t_id) %>% 
  filter(event %in% c("FIXPOINT", "EFIX") ) %>% 
  mutate(x_plotting = coalesce(fix_x, stimulus_x),
         y_plotting = coalesce(fix_y, stimulus_y),
         colour_plotting = case_when(
           stimulus_colour == "255 0 0" & event == "FIXPOINT" ~ "Stimulus",
           event == "EFIX" ~ "Fixations"
         )) %>% 
  group_by(eye,colour_plotting) %>% 
  mutate(event_nr = row_number()) %>% 
  ungroup()


p <- ggplot(data = fitts_law_plotting_data, 
            aes(x=x_plotting, y=y_plotting, colour = colour_plotting, 
                label = event_nr)) + 
  geom_point() +
  geom_path() +
  facet_wrap(~eye)

ggplotly(p, tooltip = c("label", "colour_plotting"))


# Smooth Pursuit ----
special_participants <- c(87, 89, 93, 96, 103, 105, 109, 117, 118, 119, 120, 127, 128, 141)

smooth_pursuits_raw <- read_parquet("data/processed/SMOOTH_PURSUITS.pq") %>% 
  filter(!participant_id %in% special_participants)


## Cleaning ----


smooth_pursuits_right <- smooth_pursuits_raw %>% 
  filter(!(event %in% c("SSACC", "SFIX"))) %>% 
  filter(eye %in% c(NA, "R")) %>% 
  mutate(m_eye = "R") %>% 
  mutate(m_time = coalesce(time, end_time)) %>% 
  arrange(participant_id, trial_id, m_time) %>% 
  group_by(participant_id, trial_id) %>% 
  fill(c(shape, speed), .direction = "downup") %>% 
  mutate(m_stand_time = m_time - min(m_time)) %>% 
  mutate(m_stand_start_time = start_time - min(m_time)) %>% 
  mutate(m_stand_end_time = end_time - min(m_time)) %>% 
  ungroup() %>% 
  select(experiment
         , participant_id
         , trial_id
         , stand_time=m_stand_time
         , eye = m_eye
         , event
         , shape
         , speed
         , stimulus_colour = colour
         , stimulus_x
         , stimulus_y
         , fix_x = x
         , fix_y = y
         , sacc_start_x = start_x
         , sacc_start_y = start_y
         , sacc_end_x = end_x
         , sacc_end_y = end_y
         , stand_start_time = m_stand_start_time 
         , stand_end_time = m_stand_end_time
         , avg_pupil_size
         , peak_velocity
         , amplitude
         , duration)



smooth_pursuits_left <- smooth_pursuits_raw %>% 
  filter(!(event %in% c("SSACC", "SFIX"))) %>% 
  filter(eye %in% c(NA, "L")) %>% 
  mutate(m_eye = "L") %>% 
  mutate(m_time = coalesce(time, end_time)) %>% 
  arrange(participant_id, trial_id, m_time) %>% 
  group_by(participant_id, trial_id) %>% 
  fill(c(shape, speed), .direction = "downup") %>% 
  mutate(m_stand_time = m_time - min(m_time)) %>% 
  mutate(m_stand_start_time = start_time - min(m_time)) %>% 
  mutate(m_stand_end_time = end_time - min(m_time)) %>% 
  ungroup() %>% 
  select(experiment
         , participant_id
         , trial_id
         , stand_time=m_stand_time
         , eye = m_eye
         , event
         , shape
         , speed
         , stimulus_colour = colour
         , stimulus_x
         , stimulus_y
         , fix_x = x
         , fix_y = y
         , sacc_start_x = start_x
         , sacc_start_y = start_y
         , sacc_end_x = end_x
         , sacc_end_y = end_y
         , stand_start_time = m_stand_start_time 
         , stand_end_time = m_stand_end_time
         , avg_pupil_size
         , peak_velocity
         , amplitude
         , duration)

smooth_pursuits <- smooth_pursuits_left %>% 
  bind_rows(smooth_pursuits_right)

## Visualisation ----

p_id <- 396
t_id <- 4

smooth_pursuits_plotting <- smooth_pursuits %>% 
  filter(participant_id == p_id , trial_id == t_id) %>% 
  filter(event %in% c("EFIX", "FIXPOINT") ) %>% 
  mutate(x_plotting = coalesce(fix_x, stimulus_x),
         y_plotting = coalesce(fix_y, stimulus_y),
         colour_plotting = case_when(
           event == "EFIX" ~ "Fixations",
           stimulus_colour == "255 255 255" & event == "FIXPOINT" ~ "Stimulus"
         )) %>% 
  group_by(eye,colour_plotting) %>% 
  mutate(event_nr = row_number()) %>% 
  ungroup()


p <- ggplot(data = smooth_pursuits_plotting, 
            aes(x=x_plotting, y=y_plotting, colour = colour_plotting, 
                label = event_nr)) + 
  geom_point() +
  geom_path() +
  xlim(0,1920) +
  ylim(0,1080)+
  facet_wrap(~eye)

ggplotly(p, tooltip = c("label", "colour_plotting"))



# Shapes ----

shapes_raw <- read_parquet("data/processed/SHAPES.pq") %>% 
  filter(!participant_id %in% special_participants)

## Cleaning ----

shapes_right <- shapes_raw %>% 
  filter(!(event %in% c("SSACC", "SFIX"))) %>% 
  filter(eye %in% c(NA, "R")) %>% 
  mutate(m_eye = "R") %>% 
  mutate(m_time = coalesce(time, end_time)) %>% 
  arrange(participant_id, trial_id, m_time) %>% 
  group_by(participant_id, trial_id) %>% 
  fill(c(shape), .direction = "downup") %>% 
  mutate(m_stand_time = m_time - min(m_time)) %>% 
  mutate(m_stand_start_time = start_time - min(m_time)) %>% 
  mutate(m_stand_end_time = end_time - min(m_time)) %>% 
  ungroup() %>% 
  select(experiment
         , participant_id
         , trial_id
         , stand_time=m_stand_time
         , eye = m_eye
         , event
         , shape
         , stimulus_colour = colour
         , stimulus_x
         , stimulus_y
         , fix_x = x
         , fix_y = y
         , sacc_start_x = start_x
         , sacc_start_y = start_y
         , sacc_end_x = end_x
         , sacc_end_y = end_y
         , stand_start_time = m_stand_start_time 
         , stand_end_time = m_stand_end_time
         , avg_pupil_size
         , peak_velocity
         , amplitude
         , duration)



shapes_left <- shapes_raw %>% 
  filter(!(event %in% c("SSACC", "SFIX"))) %>% 
  filter(eye %in% c(NA, "L")) %>% 
  mutate(m_eye = "L") %>% 
  mutate(m_time = coalesce(time, end_time)) %>% 
  arrange(participant_id, trial_id, m_time) %>% 
  group_by(participant_id, trial_id) %>% 
  fill(c(shape), .direction = "downup") %>% 
  mutate(m_stand_time = m_time - min(m_time)) %>% 
  mutate(m_stand_start_time = start_time - min(m_time)) %>% 
  mutate(m_stand_end_time = end_time - min(m_time)) %>% 
  ungroup() %>% 
  select(experiment
         , participant_id
         , trial_id
         , stand_time=m_stand_time
         , eye = m_eye
         , event
         , shape
         , stimulus_colour = colour
         , stimulus_x
         , stimulus_y
         , fix_x = x
         , fix_y = y
         , sacc_start_x = start_x
         , sacc_start_y = start_y
         , sacc_end_x = end_x
         , sacc_end_y = end_y
         , stand_start_time = m_stand_start_time 
         , stand_end_time = m_stand_end_time
         , avg_pupil_size
         , peak_velocity
         , amplitude
         , duration)

shapes <- shapes_left %>% 
  bind_rows(shapes_right)

## Visualisation ----

p_id <- 396
t_id <- 1

shapes_plotting <- shapes %>% 
  filter(participant_id == p_id , trial_id == t_id) %>% 
  filter(event %in% c("EFIX", "FIXPOINT") ) %>% 
  mutate(x_plotting = coalesce(fix_x, stimulus_x),
         y_plotting = coalesce(fix_y, stimulus_y),
         colour_plotting = case_when(
           event == "EFIX" ~ "Fixations",
           stimulus_colour == "255 255 255" & event == "FIXPOINT" ~ "Stimulus"
         )) %>% 
  group_by(eye,colour_plotting) %>% 
  mutate(event_nr = row_number()) %>% 
  ungroup()


p <- ggplot(data = shapes_plotting, 
            aes(x=x_plotting, y=y_plotting, colour = colour_plotting, 
                label = event_nr)) + 
  geom_point() +
  geom_path() +
  xlim(0,1920) +
  ylim(0,1080)+
  facet_wrap(~eye)

ggplotly(p, tooltip = c("label", "colour_plotting"))



# Evil bastard ----

evil_bastard_raw <- read_parquet("data/processed/EVIL_BASTARD.pq") %>% 
  filter(!participant_id %in% special_participants)


## Cleaning ----

evil_bastard_right <- evil_bastard_raw %>% 
  filter(!(event %in% c("SSACC", "SFIX"))) %>% 
  filter(eye %in% c(NA, "R")) %>% 
  mutate(m_eye = "R") %>% 
  mutate(m_time = coalesce(time, end_time)) %>% 
  arrange(participant_id, trial_id, m_time) %>% 
  group_by(participant_id, trial_id) %>% 
  fill(c(angle, speed), .direction = "downup") %>% 
  mutate(m_stand_time = m_time - min(m_time)) %>% 
  mutate(m_stand_start_time = start_time - min(m_time)) %>% 
  mutate(m_stand_end_time = end_time - min(m_time)) %>% 
  ungroup() %>% 
  select(experiment
         , participant_id
         , trial_id
         , stand_time=m_stand_time
         , eye = m_eye
         , event
         , angle
         , speed
         , stimulus_colour = colour
         , stimulus_x
         , stimulus_y
         , fix_x = x
         , fix_y = y
         , sacc_start_x = start_x
         , sacc_start_y = start_y
         , sacc_end_x = end_x
         , sacc_end_y = end_y
         , stand_start_time = m_stand_start_time 
         , stand_end_time = m_stand_end_time
         , avg_pupil_size
         , peak_velocity
         , amplitude
         , duration)



evil_bastard_left <- evil_bastard_raw %>% 
  filter(!(event %in% c("SSACC", "SFIX"))) %>% 
  filter(eye %in% c(NA, "L")) %>% 
  mutate(m_eye = "L") %>% 
  mutate(m_time = coalesce(time, end_time)) %>% 
  arrange(participant_id, trial_id, m_time) %>% 
  group_by(participant_id, trial_id) %>% 
  fill(c(angle, speed), .direction = "downup") %>% 
  mutate(m_stand_time = m_time - min(m_time)) %>% 
  mutate(m_stand_start_time = start_time - min(m_time)) %>% 
  mutate(m_stand_end_time = end_time - min(m_time)) %>% 
  ungroup() %>% 
  select(experiment
         , participant_id
         , trial_id
         , stand_time=m_stand_time
         , eye = m_eye
         , event
         , angle
         , speed
         , stimulus_colour = colour
         , stimulus_x
         , stimulus_y
         , fix_x = x
         , fix_y = y
         , sacc_start_x = start_x
         , sacc_start_y = start_y
         , sacc_end_x = end_x
         , sacc_end_y = end_y
         , stand_start_time = m_stand_start_time 
         , stand_end_time = m_stand_end_time
         , avg_pupil_size
         , peak_velocity
         , amplitude
         , duration)

evil_bastard <- evil_bastard_left %>% 
  bind_rows(evil_bastard_right)


evil_bastard %>% 
  count(stimulus_colour)

## Visualisation ----

p_id <- 106
t_id <- 2

evil_bastard_plotting <- evil_bastard %>% 
  filter(participant_id == p_id , trial_id == t_id) %>% 
  filter(event %in% c("EFIX", "FIXPOINT") ) %>% 
  mutate(x_plotting = coalesce(fix_x, stimulus_x),
         y_plotting = coalesce(fix_y, stimulus_y),
         colour_plotting = case_when(
           event == "EFIX" ~ "Fixations",
           stimulus_colour == "0 0 255" & event == "FIXPOINT" ~ "Stimulus - Blue",
           stimulus_colour == "255 0 0" & event == "FIXPOINT" ~ "Stimulus - Red",
           event == 'FIXPOINT' ~ 'Stimulus'
         )) %>% 
  group_by(eye,colour_plotting) %>% 
  mutate(event_nr = row_number()) %>% 
  ungroup()


p <- ggplot(data = evil_bastard_plotting, 
            aes(x=x_plotting, y=y_plotting, colour = colour_plotting, 
                label = event_nr)) + 
  geom_point() +
  geom_path() +
  xlim(0,1920) +
  ylim(0,1080)+
  facet_wrap(~eye)

ggplotly(p, tooltip = c("label", "colour_plotting"))

