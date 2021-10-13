# # mpirun -np 6 python -u train_HER.py --env-name='FetchSlide' --n-epochs=100
# # python collect_distance_data.py --episodes=50000 --env-name=FetchPickAndPlace --agent-location=saved_models/her_FetchPickAndPlace.pkl --max-steps=35
# # python collect_distance_data.py --episodes=50000 --env-name=FetchSlide --agent-location=saved_models/her_FetchSlide.pkl --max-steps=35
# # python train_distance_function.py  --batch-size=1024  --cuda --env-name=FetchPickAndPlace --hidden-size=256 --epoch=100 --dropout-rate=.5
# # python train_distance_function.py  --batch-size=1024  --cuda --env-name=FetchSlide --hidden-size=256 --epoch=100 --dropout-rate=.5


# # mpirun -np 6 python -u train_HER.py --env-name='FetchReach' --n-epochs=5 
# # mpirun -np 6 python -u train_HER.py --env-name='FetchReach' --n-epochs=5 --p2p
# mpirun -np 6 python -u train_HER.py --env-name='FetchPush' --n-epochs=10 --p2p
# mpirun -np 6 python -u train_HER.py --env-name='FetchPickAndPlace' --n-epochs=20
# mpirun -np 6 python -u train_HER.py --env-name='FetchPickAndPlace' --n-epochs=20 --p2p

# python main.py FetchSlide rl-then-sst
# python main.py FetchSlide psst rl-rrt
# mpirun -np 6 python -u train_HER.py --env-name='FetchPush' --n-epochs=10
# mpirun -np 6 python -u train_HER.py --env-name='FetchPush' --n-epochs=10 --p2p
python main.py FetchPush rl-rrt psst rl-then-sst 
python main.py FetchPickAndPlace psst rl-then-sst
# python main.py FetchSlide rl-rrt

# mpirun -np 6 python -u train_HER_mod.py --env-name=MultiGoalEnvironment --n-epochs=10
# mpirun -np 6 python -u train_HER_mod.py --env-name=MultiGoalEnvironment --n-epochs=10 --p2p

# python main.py GymMomentum stable-sparse-rrt psst rl
# python main.py GymMomentumShift stable-sparse-rrt psst rl #rl-then-sst rl-rrt
# python main.py GymMomentumShift2 stable-sparse-rrt psst rl #rl-then-sst rl-rrt
# python main.py GymMomentumShift3 stable-sparse-rrt psst rl #rl-then-sst rl-rrt
# python main.py GymMomentumShift4 stable-sparse-rrt psst rl #rl-then-sst rl-rrt

# python collect_distance_data_p2p.py --env-name=Asteroids --episodes=10000 --agent-location=saved_models/her_FetchReach_p2p.pkl
# python train_distance_function.py --env-name=Asteroids --epochs=200 --p2p --agent-location=saved_models/her_FetchReach.pkl
# mpirun -np 6 python -u train_HER_mod.py --env-name=Asteroids --n-epochs=10
# mpirun -np 6 python -u train_HER_mod.py --env-name=Asteroids --n-epochs=10 --p2p

# python main.py GymAsteroids stable-sparse-rrt psst rl
# python main.py GymAsteroidsShift stable-sparse-rrt psst rl #rl-then-sst rl-rrt
# python main.py GymAsteroidsShift2 stable-sparse-rrt psst rl #rl-then-sst rl-rrt
# python main.py GymAsteroidsShift3 stable-sparse-rrt psst rl #rl-then-sst rl-rrt
# python main.py GymAsteroidsShift4 stable-sparse-rrt psst rl #rl-then-sst rl-rrt

# python main.py GymAsteroidsShift rl-rrt
# python main.py GymAsteroidsShift2 rl-rrt
# python main.py GymAsteroidsShift3 rl-rrt
# python main.py GymAsteroidsShift4 rl-rrt

# mpirun -np 6 python -u train_HER.py --env-name='FetchSlide' --n-epochs=20
# mpirun -np 6 python -u train_HER.py --env-name='FetchSlide' --n-epochs=20 --p2p



# # mpirun -np 6 python -u train_HER.py --env-name='HandReach' --n-epochs=10
# # mpirun -np 6 python -u train_HER.py --env-name='HandReach' --n-epochs=10 --p2p
# # mpirun -np 6 python -u train_HER.py --env-name='HandManipulateBlock-v0' --n-epochs=100
# # mpirun -np 6 python -u train_HER.py --env-name='HandManipulateEgg-v0' --n-epochs=100
# # mpirun -np 6 python -u train_HER.py --env-name='HandManipulatePen-v0' --n-epochs=100

# # python collect_distance_data_p2p.py --env-name=FetchPickAndPlace --episodes=10000 --agent-location=saved_models/her_FetchPickAndPlace_p2p.pkl
# # python train_distance_function.py --env-name=FetchPickAndPlace --epochs=200 --p2p --agent-location=saved_models/her_FetchPickAndPlace.pkl
# # python collect_distance_data_p2p.py --env-name=FetchPush --episodes=10000 --agent-location=saved_models/her_FetchPush_p2p.pkl
# # python train_distance_function.py --env-name=FetchPush --epochs=200 --p2p --agent-location=saved_models/her_FetchPush.pkl
# # python collect_distance_data_p2p.py --env-name=FetchReach --episodes=10000 --agent-location=saved_models/her_FetchReach_p2p.pkl
# # python train_distance_function.py --env-name=FetchReach --epochs=200 --p2p --agent-location=saved_models/her_FetchReach.pkl
# python collect_distance_data_p2p.py --env-name=FetchSlide --episodes=10000 --agent-location=saved_models/her_FetchReach_p2p.pkl
# python train_distance_function.py --env-name=FetchSlide --epochs=200 --p2p --agent-location=saved_models/her_FetchReach.pkl
# # python collect_distance_data_p2p.py --env-name=HandReach --episodes=10000 --agent-location=saved_models/her_FetchReach_p2p.pkl
# # python train_distance_function.py --env-name=HandReach --epochs=200 --p2p --agent-location=saved_models/her_FetchReach.pkl

# # mpirun -np 6 python -u train_HER_mod.py --env-name='HandReach-v0' --n-epochs=50
# # mpirun -np 6 python -u train_HER_mod.py --env-name='HandReach-v0' --n-epochs=50 --p2p
# # mpirun -np 6 python -u train_HER.py --env-name='HandManipulateBlock-v0' --n-epochs=100
# # mpirun -np 6 python -u train_HER.py --env-name='HandManipulateEgg-v0' --n-epochs=100
# # mpirun -np 6 python -u train_HER.py --env-name='HandManipulatePen-v0' --n-epochs=100

# # mpirun -np 6 python -u train_HER_mod.py --env-name='State-Based-Navigation-2d-Map4-Goal0-v0' --n-epochs=10 --gamma=.995
# # mpirun -np 6 python -u train_HER_mod.py --env-name='Limited-Range-Based-Navigation-2d-Map4-Goal0-v0' --n-epochs=4 --gamma=.995


# # python main.py FetchReach psst rl-rrt
# # python main.py FetchPush psst 
# # python main.py FetchPush rl-rrt
# # python main.py FetchPickAndPlace psst 
# # python main.py FetchPickAndPlace rl-rrt
# python main.py FetchSlide psst 
# python main.py FetchSlide rl-rrt
# # python main.py HandReach psst 
# # python main.py HandReach rl-rrt

# # python main.py FetchReach rl-then-sst 
# # python main.py FetchReach pgdsst
# # python main.py FetchPush rl-then-sst 
# # python main.py FetchPush pgdsst
# # python main.py FetchPickAndPlace rl-then-sst 
# # python main.py FetchPickAndPlace pgdsst
# python main.py FetchSlide rl-then-sst 
# python main.py FetchSlide pgdsst
# # python main.py HandReach rl-then-sst 
# # python main.py HandReach pgdsst


# mpirun -np 6 python -u train_HER.py --env-name='Limited-Range-Based-Navigation-2d-Map6-Goal0-v0' --n-epochs=5 --gamma=.99
# mpirun -np 6 python -u train_HER.py --env-name='Limited-Range-Based-Navigation-2d-Map6-Goal0-v0' --n-epochs=5 --gamma=.99 --p2p

# python collect_distance_data_p2p.py --env-name=Limited-Range-Based-Navigation-2d-Map0-Goal0-v0 --episodes=10000 --agent-location=saved_models/her_Limited-Range-Based-Navigation-2d-Map4-Goal0-v0_p2p.pkl
# python train_distance_function.py --env-name=Limited-Range-Based-Navigation-2d-Map0-Goal0-v0 --epochs=200 --p2p --agent-location=saved_models/her_Limited-Range-Based-Navigation-2d-Map4-Goal0-v0.pkl

# python main.py GymObstacle2DLidar stable-sparse-rrt 
# python main.py GymObstacle2DLidar rl 
# python main.py GymObstacle2DLidar psst 
# python main.py GymObstacle2DLidar rl-rrt 
# python main.py GymObstacle2DLidar rl-then-sst 
# python main.py GymObstacle2DLidar pgdsst
