# mpirun -np 6 python -u train_HER.py --env-name='FetchSlide' --n-epochs=100
# python collect_distance_data.py --episodes=50000 --env-name=FetchPickAndPlace --agent-location=saved_models/her_FetchPickAndPlace.pkl --max-steps=35
# python collect_distance_data.py --episodes=50000 --env-name=FetchSlide --agent-location=saved_models/her_FetchSlide.pkl --max-steps=35
# python train_distance_function.py  --batch-size=1024  --cuda --env-name=FetchPickAndPlace --hidden-size=256 --epoch=100 --dropout-rate=.5
# python train_distance_function.py  --batch-size=1024  --cuda --env-name=FetchSlide --hidden-size=256 --epoch=100 --dropout-rate=.5


# mpirun -np 6 python -u train_HER.py --env-name='FetchReach' --n-epochs=10 
# mpirun -np 6 python -u train_HER.py --env-name='FetchReach' --n-epochs=10 --p2p
# mpirun -np 6 python -u train_HER.py --env-name='FetchPush' --n-epochs=50 
# mpirun -np 6 python -u train_HER.py --env-name='FetchPush' --n-epochs=50 --p2p
# mpirun -np 6 python -u train_HER.py --env-name='FetchPickAndPlace' --n-epochs=200
# mpirun -np 6 python -u train_HER.py --env-name='FetchPickAndPlace' --n-epochs=200 --p2p
# mpirun -np 6 python -u train_HER.py --env-name='FetchSlide' --n-epochs=300
# mpirun -np 6 python -u train_HER.py --env-name='FetchSlide' --n-epochs=300 --p2p

# python collect_distance_data_p2p.py --env-name=FetchPickAndPlace --episodes=10000 --agent-location=saved_models/her_FetchPickAndPlace_p2p.pkl
# python train_distance_function.py --env-name=FetchPickAndPlace --epochs=200 --p2p --agent-location=saved_models/her_FetchPickAndPlace.pkl
# python collect_distance_data_p2p.py --env-name=FetchPush --episodes=10000 --agent-location=saved_models/her_FetchPush_p2p.pkl
# python train_distance_function.py --env-name=FetchPush --epochs=200 --p2p --agent-location=saved_models/her_FetchPush.pkl
# python collect_distance_data_p2p.py --env-name=FetchReach --episodes=10000 --agent-location=saved_models/her_FetchReach_p2p.pkl
# python train_distance_function.py --env-name=FetchReach --epochs=200 --p2p --agent-location=saved_models/her_FetchReach.pkl
# python collect_distance_data_p2p.py --env-name=FetchSlide --episodes=10000 --agent-location=saved_models/her_FetchReach_p2p.pkl
# python train_distance_function.py --env-name=FetchSlide --epochs=200 --p2p --agent-location=saved_models/her_FetchReach.pkl


# python train_HER.py --env-name='Asteroids' --n-epochs=10
# python train_HER.py --env-name='AsteroidsVelGoal' --n-epochs=50
python train_HER.py --env-name='MultiGoalEnvironment' --n-epochs=50
# python train_HER.py --env-name='MultiGoalEnvironmentVelGoal' --n-epochs=50

python main.py GymMomentum rl stable-sparse-rrt psst pgdsst gdsst
# python main.py GymAsteroids rl stable-sparse-rrt psst pgdsst gdsst

# python main.py GymMomentumShift stable-sparse-rrt gdsst psst pgdsst
# python main.py GymAsteroidsShift stable-sparse-rrt gdsst psst pgdsst

# python main.py FetchReach rl
# python main.py FetchPush  rl
# python main.py FetchPickAndPlace rl
# # python main.py FetchSlide rl

# python main.py FetchPush  stable-sparse-rrt pgdsst psst 
# python main.py FetchPickAndPlace stable-sparse-rrt pgdsst psst 
# python main.py FetchPickAndPlace stable-sparse-rrt pgdsst psst 
# python main.py FetchSlide pgdsst


# python main.py FetchReach gdsst 
# python main.py FetchPush  gdsst 
# python main.py FetchPickAndPlace gdsst 
# python main.py FetchSlide gdsst 

# python main.py FetchReach psst 
# python main.py FetchPush  psst 
# python main.py FetchPickAndPlace psst 
# python main.py FetchSlide psst

# python main.py GymMomentumShift rl stable-sparse-rrt gdsst psst pgdsst
# python main.py GymAsteroidsShift rl stable-sparse-rrt gdsst psst pgdsst

# python main.py GymMomentumShift gdsst pgdsst
# python main.py GymAsteroidsShift gdsst pgdsst