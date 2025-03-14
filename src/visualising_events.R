library(tidyverse)
library(openxlsx)
library(arrow)
library(ggExtra)
library(plotly)

setwd("C:/Users/idwe/Documents/Github/Analyzing-concussions-through-eyetracking-measurements")

special_participants <- c(87, 89, 93, 96, 103, 105, 109, 117, 118, 119, 120, 127, 128, 141)
# Anti saccade ----

anti_saccade_raw <- read_parquet("data/processed/ANTI_SACCADE.pq")

## Cleaning ----

anti_saccade <- anti_saccade_raw %>% 
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


anti_saccade


## Visualising ----

p_id <- 237
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
  group_by(colour_plotting) %>% 
  mutate(event_nr = row_number())



ggplot() + 
  ggforce::geom_link2(data = anti_saccade_plotting_data %>% filter(colour_plotting %in% c("Fixations - Post stimulus", "Fixations - Pre stimulus"))
                      , aes(x=x_plotting, y=y_plotting, colour = colour_plotting, alpha=stat(index)),
                      lineend = 'round', n = 10) +
  geom_point(data = anti_saccade_plotting_data %>% mutate(duration = if_else(is.na(duration), 100, duration)),
             aes(x=x_plotting, y=y_plotting, colour = colour_plotting, size=duration/4)) 

# Reactions ----


reactions_raw <- read_parquet("data/processed/REACTION.pq") %>% 
  filter(!(participant_id %in% special_participants))



reactions <- reactions_raw %>% 
  mutate(time = coalesce(time, end_time)) %>% 
  arrange(participant_id, trial_id, time) 




